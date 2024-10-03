from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
import os
import torch.nn as nn
import torch.nn.functional as F

app = FastAPI()

class CNNClassificationModel(nn.Module):
    """
    CNN (convolutional neural network) based classification mode
    """

    def __init__(self, num_classes=2):
        super(CNNClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1)
        self.fc1 = nn.Linear(128 * 27 * 22, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 27 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Loading the pre-trained model
model = CNNClassificationModel()
model.load_state_dict(torch.load('best.pt', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Defining transformations for the input image
transform = transforms.Compose([
    transforms.Resize((218, 178)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict/")
# Since model classifies whether the person is wearing glasses or not, it needs a picture
async def predict_image(file: UploadFile = File(...)):
    # Reading the file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Preprocessing the image
    image = transform(image).unsqueeze(0)
    
    # Making prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return {"is_wearing_glasses": bool(predicted.item())}