import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


if __name__ == '__main__':
    # 여기에 본래의 코드를 추가
# Hyperparameters
    batch_size = 512
    num_classes = 10
    num_epochs = 70
    learning_rate = 0.001
    lr_weight_decay = 0.95
    dropout = 0.5  # Dropout rate

    # Check if GPU is available
    device = torch.device("cuda")

    # Load CIFAR-10 Dataset and move to GPU
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Define CNN Model and move to GPU
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

    model = CNN(dropout).to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters())

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to GPU
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)  # Move data to GPU
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('Epoch [{}/{}], Test Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, accuracy))

    print("Final Test Accuracy: {:.2f}%".format(accuracy))

    # Training loop

    torch.save(model.state_dict(), 'cifar-10.pth')

    print("Final Test Accuracy: {:.2f}%".format(accuracy))
