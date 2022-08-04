import random
import shutil
from app.ml import predict
from fastapi import FastAPI, UploadFile


app = FastAPI()


@app.post("/ml/")
def make_prediction(audio: UploadFile):
    dir = f"./audios/aud{random.randrange(start=0, stop=10)}.wav"
    with open(dir, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)
    
    prediction = predict(dir)
    return prediction
