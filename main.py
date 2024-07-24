from dataloader import load_data
from models.ESAU_net import ESAU
from train import train_model

# Get the dataloaders
train_loader, val_loader, test_loader = load_data(train_dir='dataset/train', val_dir='dataset/val', test_dir='dataset/test', batch_size=8)

# Model Initialization
model = ESAU()

# Train the model
train_model(model, train_loader, val_loader)