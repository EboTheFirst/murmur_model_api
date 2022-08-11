import random
import shutil
from app.ml import predict
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ml/")
def make_prediction(audio: UploadFile):
    dir = f"./audios/aud{random.randrange(start=0, stop=10)}.wav"
    with open(dir, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    prediction = predict(dir)
    return prediction
