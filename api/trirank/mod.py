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

tenant_id_to_model: Dict[int, ScrutableTriRank] = {}
tenant_id_to_train_set: Dict[int, Dataset] = {}

_container_client: Optional[ContainerClient] = None

def _get_container_client() -> ContainerClient:
    """Lazily create (and cache) the blob container client.

    Credential/network errors (e.g. missing `az login` when running
    locally) are only surfaced here, at the point of use, rather than at
    import time or app startup.
    """
    global _container_client
    if _container_client is None:
        try:
            credential = DefaultAzureCredential()
            blob_service_client = BlobServiceClient(account_url, credential=credential)
            _container_client = blob_service_client.get_container_client(container=container_name)
            # Force a lightweight call so auth failures surface immediately
            # instead of lazily on some later, unrelated call.
            _container_client.get_container_properties()
        except Exception as e:
            _container_client = None
            raise RuntimeError(
                "Azure Blob Storage is unavailable. If running locally, make sure "
                f"you are authenticated (e.g. run `az login`). Original error: {e}"
            ) from e
    return _container_client

def _load_model_from_blob(tenant_id: int):
    container_client = _get_container_client()

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
    try:
        container_client = _get_container_client()
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
    except Exception as e:
        print(f"[trirank] skipping startup model preload, Azure Blob Storage unavailable: {e}")
    yield

app.router.lifespan_context = lifespan

class RecommendationRequest(BaseModel):
    tenant_id: int
    uid: str
    k: int
    # (aspect_name, weight) pairs with weight already in [0, 1]. Callers own the
    # mapping from whatever scale their UI uses onto aspect weights, because they
    # are also the ones who decide the aspect vocabulary when building the
    # training data - see the backend's TriRankAspects. This endpoint used to
    # accept a raw 1-5 score and apply an exponential curve to it, which both
    # duplicated that knowledge and, being monotonic, could only ever express
    # "I care about this aspect more", never "I want the opposite end of it".
    preferences: Optional[List[Tuple[str, float]]] = None
    remove_seen: Optional[bool] = False
    # Weight of the review-derived prior when blending in `preferences`.
    a0_prior_weight: Optional[float] = None

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
    if req.a0_prior_weight is not None:
        temp_model.a0_prior_weight = req.a0_prior_weight

    unknown_aspects = []
    for aspect_name, weight in req.preferences:
        aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name, -1)
        if aspect_idx == -1:
            unknown_aspects.append(aspect_name)
            continue
        temp_model.a0_overrides[aspect_idx] = min(max(float(weight), 0.0), 1.0)

    # An aspect name the model has never seen used to be skipped silently, so a
    # model trained against an older aspect vocabulary would quietly ignore every
    # preference and return unpersonalised results that look plausible. Since the
    # vocabulary is baked into the saved trainset, that is exactly what happens if
    # the dataset is not re-exported and retrained after it changes, so say so.
    if unknown_aspects:
        print(f"[trirank] tenant {req.tenant_id}: unknown aspects ignored: {unknown_aspects}")
    if not temp_model.a0_overrides:
        raise HTTPException(
            status_code=400,
            detail=(
                f"None of the requested aspects exist in the model for tenant {req.tenant_id}. "
                f"The model was most likely trained before the aspect vocabulary changed and needs "
                f"re-exporting and retraining. Unknown aspects: {unknown_aspects}"
            ),
        )

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
    try:
        _load_model_from_blob(req.tenant_id)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
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
    try:
        _load_model_from_blob(tenant_id)
        print(f"[trirank train] model reloaded for tenant {tenant_id}")
    except Exception as e:
        print(f"[trirank train] failed to reload model for tenant {tenant_id}: {e}")
