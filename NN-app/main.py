import torch
import torch.nn.functional as F
from torchvision import transforms

import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from helper_lib.model import get_model
from helper_lib.utils import get_device

app = FastAPI(title="CIFAR-10 Image Classifier API")

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
DEVICE = get_device()
MODEL_PATH = "cifar10_cnn.pth"

model = get_model("CNN")
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def transform_image(image_bytes: bytes):
    """
    Takes image bytes, applies the necessary transformations,
    and returns a tensor ready for the model.
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image bytes from the uploaded file
        img_bytes = await file.read()

        # Preprocess the image
        tensor = transform_image(img_bytes)
        tensor = tensor.to(DEVICE)

        # Make a prediction
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_prob, top_class_index = probabilities.topk(1, dim=1)

            predicted_class = CIFAR10_CLASSES[top_class_index.item()]
            confidence = top_prob.item()

        # FastAPI automatically converts dictionaries to JSON
        return {
            "prediction": predicted_class,
            "confidence": f"{confidence:.4f}"
        }
    except Exception as e:
        return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)