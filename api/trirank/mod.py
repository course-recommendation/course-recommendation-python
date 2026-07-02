from api.mod import app
from api.trirank.scrutable import ScrutableTriRank, to_scrutable
from cornac.models import TriRank
from cornac.data import Dataset
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import os
from typing import List, Tuple, Optional, Dict
import copy
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential
import asyncio
import sys

MODEL_DIR = "./model/trirank"
os.makedirs(MODEL_DIR, exist_ok=True)

storage_account_name = "stcourserecom"
container_name = "model"
account_url = f"https://{storage_account_name}.blob.core.windows.net"
default_credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url, credential=default_credential)
container_client = blob_service_client.get_container_client(container=container_name)

tenant_id_to_model: Dict[int, ScrutableTriRank] = {}
tenant_id_to_train_set: Dict[int, Dataset] = {}

def _load_model_from_blob(tenant_id: int):
    model_blob = f"{tenant_id}.model.pkl"
    trainset_blob = f"{tenant_id}.model.pkl.trainset"
    model_path = os.path.join(MODEL_DIR, model_blob)
    trainset_path = os.path.join(MODEL_DIR, trainset_blob)

    print(f"\nDownloading blob to {model_path}")
    with open(file=model_path, mode="wb") as f:
        f.write(container_client.download_blob(model_blob).readall())

    print(f"\nDownloading blob to {trainset_path}")
    with open(file=trainset_path, mode="wb") as f:
        f.write(container_client.download_blob(trainset_blob).readall())

    tenant_id_to_model[tenant_id] = to_scrutable(TriRank.load(model_path))
    tenant_id_to_train_set[tenant_id] = Dataset.load(trainset_path)

@asynccontextmanager
async def lifespan(app: FastAPI):
    blobs = container_client.list_blobs()
    tenant_ids = set()
    for blob in blobs:
        name = blob.name
        if name.endswith(".model.pkl"):
            prefix = name[: -len(".model.pkl")]
            if prefix.isdigit():
                tenant_ids.add(int(prefix))
    for tenant_id in tenant_ids:
        try:
            _load_model_from_blob(tenant_id)
        except Exception as e:
            print(f"[trirank] failed to load model for tenant {tenant_id}: {e}")
    yield

app.router.lifespan_context = lifespan

class RecommendationRequest(BaseModel):
    tenant_id: int
    uid: str
    k: int
    preferences: Optional[List[Tuple[str, float]]] = None
    remove_seen: Optional[bool] = False

# Preference scale bounds and the curve used to convert a raw [MIN_SCORE,
# MAX_SCORE] preference into an a0_overrides weight in [0, 1].
#
# A linear mapping (score / MAX_SCORE) treats every one-point step as
# equally significant, so a 4 vs 5 choice barely differs from a 1 vs 2
# choice. Using an exponential curve instead makes higher scores pull
# away from lower ones multiplicatively: each +1 on the scale multiplies
# the weight by PREFERENCE_CURVE_BASE, so with base=2.0 a score of 5 ends
# up 2**4 = 16x the weight of a score of 1, rather than just 5x.
# Tune PREFERENCE_CURVE_BASE to taste: 1.0 reduces to linear, higher
# values (e.g. 3.0) make only the top of the scale matter at all.
MIN_SCORE = 1.0
MAX_SCORE = 5.0
PREFERENCE_CURVE_BASE = 3.0

def _score_to_weight(score: float) -> float:
    score = min(max(score, MIN_SCORE), MAX_SCORE)
    exponent = score - MIN_SCORE
    max_exponent = MAX_SCORE - MIN_SCORE
    return (PREFERENCE_CURVE_BASE ** exponent) / (PREFERENCE_CURVE_BASE ** max_exponent)

@app.post("/trirank/recommendation")
async def get_recommendation(req: RecommendationRequest):
    model = tenant_id_to_model.get(req.tenant_id)
    train_set = tenant_id_to_train_set.get(req.tenant_id)
    if model is None or train_set is None:
        raise HTTPException(status_code=404, detail=f"No model for tenant {req.tenant_id}")

    if not req.preferences:
        response = model.recommend(user_id=req.uid, k=req.k, remove_seen=req.remove_seen, train_set=train_set)
        return {"recommendations": response}

    temp_model = copy.copy(model)
    temp_model.a0_overrides = {}

    for aspect_name, score in req.preferences:
        aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name, -1)
        if aspect_idx == -1:
            continue
        # Map the incoming [1, 5] preference scale to a [0, 1] weight,
        # non-linearly so higher choices dominate lower ones instead of
        # scaling proportionally. See _score_to_weight for the curve.
        temp_model.a0_overrides[aspect_idx] = _score_to_weight(score)

    response = temp_model.recommend(user_id=req.uid, k=req.k, remove_seen=req.remove_seen, train_set=train_set)
    return {"recommendations": response}

@app.get("/trirank/topk-aspect-of-item")
async def topk_aspect_of_item(tenant_id: int, item_id: str, k: int):
    model = tenant_id_to_model.get(tenant_id)
    train_set = tenant_id_to_train_set.get(tenant_id)
    if model is None or train_set is None:
        raise HTTPException(status_code=404, detail=f"No model for tenant {tenant_id}")

    item_idx = train_set.iid_map.get(item_id, -1)
    aspect_score_of_item = model.X[item_idx].toarray().squeeze()
    id_to_aspect = {idx: name for name, idx in train_set.sentiment.aspect_id_map.items()}

    aspect_name_and_score_of_item = [(id_to_aspect.get(i), float(aspect_score_of_item[i]))
                                     for i in range(len(aspect_score_of_item))]
    aspect_name_and_score_of_item.sort(reverse=True, key=lambda x: x[1])
    return aspect_name_and_score_of_item[:k]

@app.get("/trirank/score-of-aspect-to-user")
async def score_of_aspect_to_user(tenant_id: int, uid: str, aspect_name: str):
    model = tenant_id_to_model.get(tenant_id)
    train_set = tenant_id_to_train_set.get(tenant_id)
    if model is None or train_set is None:
        raise HTTPException(status_code=404, detail=f"No model for tenant {tenant_id}")

    user_idx = train_set.uid_map.get(uid, -1)
    aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name, -1)
    return model.Y[user_idx, aspect_idx]

class ReloadRequest(BaseModel):
    tenant_id: int

@app.post("/trirank/reload")
async def reload_model(req: ReloadRequest):
    _load_model_from_blob(req.tenant_id)
    return {"status": "model reloaded", "tenant_id": req.tenant_id}

class TrainRequest(BaseModel):
    tenant_id: int

@app.post("/trirank/train")
async def train(req: TrainRequest):
    asyncio.create_task(_run_training(req.tenant_id))
    return {"status": "training started", "tenant_id": req.tenant_id}

async def _run_training(tenant_id: str):
    scripts = ["trirank.py", "upload_trirank.py"]
    for script in scripts:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script, str(tenant_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            print(f"[trirank train] {script} failed:\n{stderr.decode()}")
            return
        print(f"[trirank train] {script} succeeded:\n{stdout.decode()}")
    _load_model_from_blob(tenant_id)
    print(f"[trirank train] model reloaded for tenant {tenant_id}")
