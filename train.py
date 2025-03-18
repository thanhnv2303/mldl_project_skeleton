import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

data_folder = "data"
transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tiny_imagenet_dataset_train = ImageFolder(root='{data_folder}/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='{data_folder}/tiny-imagenet-200/val', transform=transform)

print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)

import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Add more layers...
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()


        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Replaces flatten + large FC layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128,200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)

        return x

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # todo...
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()

            # todo...
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy

model = CustomNet().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

best_acc = 0

# Run the training process for {num_epochs} epochs
num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(epoch, model, train_loader, criterion, optimizer)

    # At the end of each training iteration, perform a validation step
    val_accuracy = validate(model, val_loader, criterion)

    # Best validation accuracy
    best_acc = max(best_acc, val_accuracy)

print(f'Best validation accuracy: {best_acc:.2f}%')