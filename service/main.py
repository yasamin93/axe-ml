from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from PIL import Image
from transformers import AutoModelForCausalLM
import torch

app = FastAPI()
model = AutoModelForCausalLM.from_pretrained(
    "q-future/one-align",
    trust_remote_code=True,
    attn_implementation="eager",
    torch_dtype=torch.float16,
    device_map="auto",
    offload_folder="offload",
)


def get_quality_and_aesthetic_score(image):
    quality_score = model.score([image], task_="quality", input_="image")
    aesthetic_score = model.score([image], task_="aesthetics", input_="image")
    return {"quality": quality_score, "aesthetic": aesthetic_score}


class Prediction(BaseModel):
    quality: float
    aesthetic: float


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict", response_model=Prediction)
def predict(image: UploadFile):
    image = Image.open(image.file)
    return get_quality_and_aesthetic_score(image)
