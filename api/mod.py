from fastapi import FastAPI

app = FastAPI()

# Import to register routes
from api.feature_sentiments import mod as _
