from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import base64
from typing import List  # Add this import
import io
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from pydantic import BaseModel
import os
import uuid

app = FastAPI()
num_classes = 10
# 정적 파일 (CSS 및 JavaScript)을 제공하기 위한 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
# PyTorch 모델을 불러오는 부분
class CNN(nn.Module):
    def __init__(self, dropout):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out

# 모델 초기화
model = CNN(dropout=0.5)

# 사용 가능한 CPU로 모델 이동 (GPU 코드 주석 처리)
device = torch.device("cpu")
model = model.to(device)

# 모델 가중치 불러오기 
model.load_state_dict(torch.load('cifar-10.pth', map_location=device))

# 모델을 평가 모드로 설정
model.eval()


# 이미지 전처리를 위한 변환

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

class PredictionRequest(BaseModel):
    image: str  # Receive image data as a string

class PredictionResult(BaseModel):
    class_name: str
    probability: float

@app.post("/predict/", response_model=List[PredictionResult])
async def predict_drawing(drawing_data: PredictionRequest):

    # Decode base64 image data and convert to NumPy array
    image_base64 = drawing_data.image.replace("data:image/png;base64,", "")
    image_bytes = io.BytesIO(base64.b64decode(image_base64))
    image = Image.open(image_bytes)
    image = image.convert("RGB")

    
     # Now, let's save the image
    file_extension = ".png"  # You can change the extension if needed
    unique_filename = str(uuid.uuid4()) + "_1" + file_extension
    output_image_path = os.path.join("predicted_images", unique_filename)

    image.save(output_image_path)

   

    # Apply the same preprocessing transformations as defined in transform
    image = transform(image).unsqueeze(0)

    # Perform the prediction with your model
    with torch.no_grad():
        outputs = model(image)

    predicted_scores = torch.softmax(outputs, dim=1)[0]
    class_indices = torch.argsort(predicted_scores, descending=True)

    # Get the predicted class with the highest probability
    top_class_idx = torch.argmax(predicted_scores).item()
    top_class = class_labels[top_class_idx]
    top_probability = predicted_scores[top_class_idx].item()

    # Create a list of PredictionResult objects for all classes, sorted by probability
    predictions = []
    for idx in class_indices:
        class_name = class_labels[idx]
        probability = predicted_scores[idx].item()
        predictions.append({"class_name": class_name, "probability": probability})
    return predictions
   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

        
