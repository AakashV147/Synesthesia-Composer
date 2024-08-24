import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json

# Simple dataset class for loading COCO-like data
class TextImageDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        image_name = self.captions[idx]['image']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return caption, image

# Simple Text-to-Image model (you can improve this)
class TextToImageModel(nn.Module):
    def __init__(self):
        super(TextToImageModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 64 * 3)  # Assuming a 64x64 image
        )

    def forward(self, text_embedding):
        return self.fc(text_embedding).view(-1, 3, 64, 64)

# Training function
def train_text_to_image(image_dir, captions_file, output_model_path, num_epochs=10, batch_size=32, learning_rate=1e-4):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = TextImageDataset(image_dir, captions_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TextToImageModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for captions, images in dataloader:
            # Convert captions to embeddings (this is a placeholder, you need a real text encoder like BERT, GPT, etc.)
            text_embeddings = torch.randn(len(captions), 512)  # Random embeddings for now

            outputs = model(text_embeddings)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}')

    # Save model
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")

# Example usage
if __name__ == '__main__':
    image_dir = '/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/data/inputs/text_to_image/train/train2014'
    captions_file = '/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/data/inputs/text_to_image/train/captions_train2014.json'
    output_model_path = '/Users/aakashvenkatraman/Documents/GitHub/Synesthesia-Composer/models/huggingface/saved_models/text_to_image.pth'

    train_text_to_image(image_dir, captions_file, output_model_path)
