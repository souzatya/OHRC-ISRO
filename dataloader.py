import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class OHRCDataset(Dataset):
    def __init__(self, root_dir, mask_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = []
        self.mask_paths = []

        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            mask_class_dir = os.path.join(mask_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.png'):
                        img_path = os.path.join(class_dir, img_name)
                        mask_path = os.path.join(mask_class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask

# Define transformations
transform = transforms.Compose([
    transforms.Resize((1200, 1200)),  # Resize the images to a fixed size
    transforms.ToTensor()          # Convert the image to a tensor
])

def load_data(train_dir, val_dir, test_dir, batch_size):
    # Create datasets
    train_data = OHRCDataset(root_dir=os.path.join(train_dir,'images'), mask_dir=os.path.join(train_dir,'masks'), transform=transform, mask_transform=transform)
    val_data = OHRCDataset(root_dir=os.path.join(val_dir,'images'), mask_dir=os.path.join(val_dir,'masks'), transform=transform, mask_transform=transform)
    test_data = OHRCDataset(root_dir=os.path.join(test_dir,'images'), mask_dir=os.path.join(test_dir,'masks'), transform=transform, mask_transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


