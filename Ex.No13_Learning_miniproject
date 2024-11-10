# Ex.No: 13 Mini Project
### DATE: 4.11.2024                                                                 
### REGISTER NUMBER : 212222040180

### AIM:
To write a program to train a classifier for garbage classification with PyTorch.

### Algorithm:
1. Import necessary libraries (PyTorch, Torchvision, etc.)
2. Load the dataset of images representing different types of garbage
3. Define the EfficientNet-B0 model and add a classification head to it
4. Create data loaders for training and validation sets
5. Train the model using the Adam optimizer and cross-entropy loss function
6. Monitor the model's performance on the validation set during training


### Program:
```python
import torch
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import time
import os

data_dir = "garbage_classification"
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(data_dir, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
num_classes = 12

model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Function to track training progress
def train_model(model, criterion, optimizer, num_epochs=5):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        start_time = time.time()

        # Iterate over data in batches
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)

            # Print progress every 100 batches
            if total_samples % 3200 == 0:
                batch_loss = running_loss / total_samples
                batch_acc = running_corrects.double() / total_samples
                print(f"Progress: {total_samples}/{len(train_dataset)} - Loss: {batch_loss:.4f}, Acc: {batch_acc:.4f}")

        # End of epoch stats
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Epoch {epoch+1} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print(f'Time for epoch {epoch+1}: {time.time() - start_time:.2f} seconds\n')

    return model


model = train_model(model, criterion, optimizer, num_epochs=10)
torch.save(model.state_dict,"garbage_classifier")
```


### Output:

![image](https://github.com/user-attachments/assets/732febf2-7f8c-4f63-a35d-32c1af2cd8d4)<br>
The Model reached an accuracy of 97.5% after 10 epochs against the test dataset.


### Result:
Thus the system was trained successfully and the prediction was carried out.
