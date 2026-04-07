
from api.mod import app
from cornac.models import TriRank
from cornac.data import Dataset
from contextlib import asynccontextmanager
from fastapi import FastAPI


model = None
train_set = None

MODEL_PATH = "/Users/hien/Works/course-recommendation-python/save_dir/TriRank"
TRAIN_SET_PATH = "/Users/hien/Works/course-recommendation-python/save_dir/TriRank/2026-04-06_00-45-53-786624.pkl.trainset"

model: TriRank = None
train_set: Dataset = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, train_set
    model = TriRank.load(MODEL_PATH)
    train_set = Dataset.load(TRAIN_SET_PATH)
    yield

app.router.lifespan_context = lifespan

@app.post("/trirank/recommendation")
async def get_recommendation(uid: str, k: int):
  response = model.recommend(
    user_id=uid,
    k=k,
  )

  data = {
    "recommendations": response,
  }
  return data

@app.get("/trirank/topk-aspect-of-item")
async def topk_aspect_of_item(item_id: str, k: int):
  item_idx = train_set.iid_map.get(item_id)
  aspect_score_of_item = model.X[item_idx].toarray().squeeze()
  id_to_aspect = {idx: name for name, idx in train_set.sentiment.aspect_id_map.items()}

  aspect_name_and_score_of_item = [(id_to_aspect.get(i), float(aspect_score_of_item[i])) 
                        for i in range(len(aspect_score_of_item))]
  aspect_name_and_score_of_item.sort(reverse=True, key=lambda x: x[1])
  top_k_name_and_score_of_item = aspect_name_and_score_of_item[:k]
  print(top_k_name_and_score_of_item)
  return top_k_name_and_score_of_item

@app.get("/trirank/score-of-aspect-to-user")
async def score_of_aspect_to_user(uid: str, aspect_name: str):
  user_idx = train_set.uid_map.get(uid)
  aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name)
  aspect_score_of_user_to_aspect = model.Y[user_idx, aspect_idx]
  print(aspect_score_of_user_to_aspect)
  return aspect_score_of_user_to_aspect

@app.post("/trirank/update-aspect-score-of-user")
async def update_aspect_score_of_user(uid: str, aspect_name: str, new_score: float):
  user_idx = train_set.uid_map.get(uid)
  aspect_idx = train_set.sentiment.aspect_id_map.get(aspect_name)
  model.Y[user_idx, aspect_idx] = new_score
  return {"message": "Aspect score updated successfully."}
