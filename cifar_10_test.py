import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# Load the saved model
# Define your CNN model class
num_classes = 10
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

model = CNN(dropout=0.5)  # Initialize your model with the appropriate dropout rate
model.load_state_dict(torch.load('cifar-10.pth'))
model.eval()

# Define the image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# Load and preprocess the input image
image_path = 'pl2.jpg'  # Replace with the path to your input image
image = Image.open(image_path)
image = transform(image).unsqueeze(0)  # Add a batch dimension

# Make predictions
with torch.no_grad():
    outputs = model(image)

# Get class probabilities and class indices
probabilities = torch.softmax(outputs, dim=1)[0]
class_indices = torch.argsort(probabilities, descending=True)

# Define class labels for CIFAR-10
class_labels = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Print the predicted classes and their probabilities in descending order
for idx in class_indices:
    print(f'Class: {class_labels[idx]}, Probability: {probabilities[idx]:.2f}')
