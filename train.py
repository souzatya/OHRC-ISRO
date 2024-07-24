import torch
import torch.optim as optim
import os

# Dice Loss Function
def dice_loss(pred, target, smooth=1):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    
    return 1 - dice.mean()

# IoU Calculation Function
def iou(pred, target, smooth=1):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    total = (pred + target).sum(dim=2).sum(dim=2)
    union = total - intersection 
    
    IoU = (intersection + smooth) / (union + smooth)
    
    return IoU.mean()

# Training the model
def train_model(model, train_loader, val_loader, num_epochs=1, learning_rate=1e-3):
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss functions and optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists('weights'):
        os.makedirs('weights')
   
    model.train()

    # Training the autoencoder
    for epoch in range(num_epochs):

        train_loss_dice = 0
        train_loss_iou = 0

        for image, mask in train_loader:
            image = image.to(device)
            mask = mask.to(device)

            # Forward pass
            output = model(image)

            loss_dice = dice_loss(output, mask)
            loss_iou = 1 - iou(output, mask)
            loss = loss_dice + loss_iou

            train_loss_dice += loss_dice.item()
            train_loss_iou += loss_iou.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss_dice /= len(train_loader)
        train_loss_iou /= len(train_loader)

        # Validation
        model.eval()
        val_loss_dice = 0
        val_loss_iou = 0
        s = 0

        with torch.no_grad():
            for image, mask in val_loader:
                image = image.to(device)
                mask = mask.to(device)

                output = model(image)

                loss_dice = dice_loss(output, mask)
                loss_iou = 1 - iou(output, mask)
                val_loss_dice += loss_dice.item()
                val_loss_iou += loss_iou.item()
        
        val_loss_dice /= len(val_loader)
        val_loss_iou /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss (Dice): {train_loss_dice:.4f}, Training Loss (IoU): {train_loss_iou:.4f}, Validation Loss (Dice): {val_loss_dice:.4f}, Validation Loss (IoU): {val_loss_iou:.4f}')
        torch.save(model.state_dict(), os.path.join('weights', f'checkpoint_{epoch+1}.pt'))
        
    print('Training finished.')
