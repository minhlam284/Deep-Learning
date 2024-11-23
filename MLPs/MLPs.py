import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logging.basicConfig(
    filename="mlp_training_log.txt",
    level = logging.INFO,
    format = "%(asctime)s - %(message)s",
)

batch_size = 128
learning_rate = 0.001
num_epochs = 30
input_size = 32 * 32 * 3
hidden_sizes = [512, 256, 128]
output_size = 10

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root='./data', train = True, transform=transforms, download=True)
test_data = datasets.CIFAR10(root='./data', train = False, transform=transforms, download=True)

train_loader = DataLoader(train_data, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size= batch_size, shuffle=False)

class MLPs(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLPs, self).__init__()
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
model = MLPs(input_size, hidden_sizes, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging.info(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).to(device)  # Flatten images
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy on test data: {100 * correct / total:.2f}%')
    logging.info(f'Accuracy on test data: {100 * correct / total:.2f}%')