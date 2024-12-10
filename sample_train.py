import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define directories
data_dir = "/run/media/koosha/66FF-F330/Softwares_and_Data_Backup/AI_data/EYE/eye_diseases_classification"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the model input
    transforms.ToTensor(),         # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize images
])

# Load datasets
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

# Print number of images found
print(f"Found {len(train_dataset)} training images.")
print(f"Found {len(val_dataset)} validation images.")

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Print DataLoader sizes
print(f"Train Loader: {len(train_loader.dataset)} samples")
print(f"Validation Loader: {len(val_loader.dataset)} samples")

# Define a CNN model for 4 classes
class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 112 * 112, 128)  # Hidden layer
        self.fc2 = nn.Linear(128, 4)  # Output for 4 classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)  # Flatten the output
        x = F.relu(self.fc1(x))        # Apply first fully connected layer
        x = self.fc2(x)                # Apply output layer
        return x

# Initialize the model, criterion, and optimizer
model = AlzheimerCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10  # You can adjust the number of epochs
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Move to device
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize weights
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# Validation loop
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # Disable gradient calculation for validation
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)  # Move to device
        outputs = model(images)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
        total += labels.size(0)  # Update total
        correct += (predicted == labels).sum().item()  # Update correct predictions

print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Save the model
torch.save(model.state_dict(), 'leukemia_classifier.pth')
print("Model saved as leukemia_classifier.pth")
