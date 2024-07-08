import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise
from PIL import Image

# Custom Dataset
class NoisyDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        image = np.array(image)
        noisy_image = random_noise(image, mode='gaussian', var=0.1**2)
        noisy_image = Image.fromarray((noisy_image * 255).astype(np.uint8))
        if self.transform:
            noisy_image = self.transform(noisy_image)
            image = self.transform(Image.fromarray(image))
        return noisy_image, image

# Define CNN Model
class DenoiseCNN(nn.Module):
    def __init__(self):
        super(DenoiseCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Hyperparameters
batch_size = 16
learning_rate = 0.001
num_epochs = 10

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
noisy_train_dataset = NoisyDataset(train_dataset, transform=transform)
train_loader = DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)

# Model, Loss, Optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenoiseCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    for noisy_images, clean_images in train_loader:
        noisy_images, clean_images = noisy_images.to(device), clean_images.to(device)
        outputs = model(noisy_images)
        loss = criterion(outputs, clean_images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the Model
torch.save(model.state_dict(), 'denoise_cnn.pth')

# Test the Model
def denoise_image(noisy_image):
    model.eval()
    with torch.no_grad():
        noisy_image = noisy_image.to(device)
        restored_image = model(noisy_image.unsqueeze(0))
    return restored_image.squeeze(0).cpu()

# Display Results
test_image, _ = noisy_train_dataset[0]
restored_image = denoise_image(test_image)

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(test_image.permute(1, 2, 0))
ax[0].set_title('Noisy Image')
ax[1].imshow(restored_image.permute(1, 2, 0))
ax[1].set_title('Restored Image')
ax[2].imshow(train_dataset[0][0].permute(1, 2, 0))
ax[2].set_title('Original Image')
plt.show()
