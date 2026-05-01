from api.mod import app
from cornac.models import TriRank
from cornac.data import Dataset
from contextlib import asynccontextmanager
from fastapi import FastAPI
import os
from typing import List, Tuple, Optional
import copy
from pydantic import BaseModel
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.identity import DefaultAzureCredential

MODEL_PATH = "./model/trirank/model.pkl"
TRAIN_SET_PATH = "./model/trirank/model.pkl.trainset"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_SET_PATH), exist_ok=True)

storage_account_name = "stcourserecom"
container_name = "model"
account_url = f"https://{storage_account_name}.blob.core.windows.net"
default_credential = DefaultAzureCredential()
blob_service_client = BlobServiceClient(account_url, credential=default_credential)
container_client = blob_service_client.get_container_client(container= container_name) 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, train_set

    print("\nDownloading blob to " + MODEL_PATH)
    with open(file=MODEL_PATH, mode="wb") as download_file:
        download_file.write(container_client.download_blob("model.pkl").readall())

    print("\nDownloading blob to " + TRAIN_SET_PATH)
    with open(file=TRAIN_SET_PATH, mode="wb") as download_file:
        download_file.write(container_client.download_blob("model.pkl.trainset").readall())
    
    model = TriRank.load(MODEL_PATH)
    train_set = Dataset.load(TRAIN_SET_PATH)
    yield

app.router.lifespan_context = lifespan

class RecommendationRequest(BaseModel):
    uid: str
    k: int
    preferences: Optional[List[Tuple[str, float]]] = None
    remove_seen: Optional[bool] = False
@app.post("/trirank/recommendation")
async def get_recommendation(req: RecommendationRequest):
    uid = req.uid
    k = req.k
    preferences = req.preferences
    remove_seen = req.remove_seen
    if not preferences:
        response = model.recommend(user_id=uid, k=k, remove_seen=remove_seen, train_set=train_set)
        return {"recommendations": response}

    temp_model = copy.copy(model)
    temp_model.Y = model.Y.copy()

    user_idx = train_set.uid_map.get(uid, -1)

    for aspect_name, score in preferences:
        aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name, -1)
        if aspect_idx == -1:
            continue
        temp_model.Y[user_idx, aspect_idx] = score

    response = temp_model.recommend(user_id=uid, k=k, remove_seen=remove_seen, train_set=train_set)
    return {"recommendations": response}

@app.get("/trirank/topk-aspect-of-item")
async def topk_aspect_of_item(item_id: str, k: int):
  item_idx = train_set.iid_map.get(item_id, -1)
  aspect_score_of_item = model.X[item_idx].toarray().squeeze()
  id_to_aspect = {idx: name for name, idx in train_set.sentiment.aspect_id_map.items()}

  aspect_name_and_score_of_item = [(id_to_aspect.get(i), float(aspect_score_of_item[i])) 
                        for i in range(len(aspect_score_of_item))]
  aspect_name_and_score_of_item.sort(reverse=True, key=lambda x: x[1])
  top_k_name_and_score_of_item = aspect_name_and_score_of_item[:k]
  return top_k_name_and_score_of_item

@app.get("/trirank/score-of-aspect-to-user")
async def score_of_aspect_to_user(uid: str, aspect_name: str):
  user_idx = train_set.uid_map.get(uid, -1)
  aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name, -1)
  aspect_score_of_user_to_aspect = model.Y[user_idx, aspect_idx]
  return aspect_score_of_user_to_aspect
