import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast

# Get the directory of the current file
current_file_dir = os.path.dirname(__file__)

# Construct the path to the project root
project_root = os.path.abspath(os.path.join(current_file_dir, '..', '..'))

# Add the project root to the Python path
sys.path.append(project_root)

# Import the module
from app.utils.load_model import save_model

class TextImageDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Read captions from the .token file
        self.captions = []
        with open(captions_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_file, caption = parts
                    img_file = img_file.split('#')[0]
                    self.captions.append({"image": img_file, "caption": caption})
                else:
                    print(f"Skipping malformed line: {line}")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        image_path = os.path.join(self.image_dir, self.captions[idx]['image'])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return caption, image

class VQVAE(nn.Module):
    def __init__(self):
        super(VQVAE, self).__init__()
        # Encoder layers (downsample image)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        )
        # Decoder layers (upsample to image size)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Normalize pixel values to [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_text_to_image(image_dir, captions_file, output_model_path, num_epochs=10, learning_rate=1e-4):
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Load the dataset
    dataset = TextImageDataset(image_dir, captions_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model, loss function, and optimizer
    model = VQVAE()
    
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Mixed precision training
    scaler = GradScaler()

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (captions, images) in enumerate(dataloader):
            images = images.to(device)

            optimizer.zero_grad()  # Zero gradients

            # Mixed precision forward and backward pass
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, images)

            scaler.scale(loss).backward()  # Backward pass
            scaler.step(optimizer)  # Optimization step
            scaler.update()  # Update scaler

            total_loss += loss.item()

            # Add logging for every batch
            if (i + 1) % 10 == 0:  # Log every 10 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(dataloader)}')

    # Save the trained model
    save_model(model, output_model_path)

if __name__ == "__main__":
    # Define paths
    image_dir = '/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/data/inputs/text_to_image/images/flickr30k-images'
    captions_file = '/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/data/inputs/text_to_image/captions.token'
    output_model_path = '/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/models/huggingface/saved_models/text_to_image.pth'

    # Train the model
    train_text_to_image(image_dir, captions_file, output_model_path)
