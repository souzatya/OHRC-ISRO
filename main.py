from dataloader import load_data

train_loader, val_loader, test_loader = load_data(train_dir='dataset/train', val_dir='dataset/val', test_dir='dataset/test', batch_size=8)

# Example of iterating through the DataLoader
for images, masks in train_loader:
    print(images.shape, masks.shape)
    # Do something with the images and labels

# Example of iterating through the DataLoader
for images, masks in val_loader:
    print(images.shape, masks.shape)
    # Do something with the images and labels

# Example of iterating through the DataLoader
for images, masks in test_loader:
    print(images.shape, masks.shape)
    # Do something with the images and labels