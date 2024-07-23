

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
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss functions and optimizer
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists('val_results'):
        os.makedirs('val_results')
   
    model.train()

    # Training the autoencoder
    for epoch in range(num_epochs):
        for img, lbl in train_loader:
            img = img.to(device)
            lbl = lbl.to(device)
            
            ssim_map_true = ssim(img, lbl, size_average=False)

            # Forward pass
            output = model(img)
            ssim_map_pred = ssim(output, lbl, size_average=False)

            loss_mse = criterion_mse(ssim_map_pred, ssim_map_true)
            loss_mae = criterion_mae(ssim_map_pred, ssim_map_true)
            loss = loss_mse  # You can choose to optimize for either MSE or MAE
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss_mse = 0
        val_loss_mae = 0
        s = 0

        with torch.no_grad():
            for val_img, val_lbl in val_loader:
                val_img = val_img.to(device)
                val_lbl = val_lbl.to(device)

                ssim_map_true = ssim(val_img, val_lbl, size_average=False)
                output = model(val_img)
                ssim_map_pred = ssim(output, val_lbl, size_average=False)

                loss_mse = criterion_mse(ssim_map_pred, ssim_map_true)
                loss_mae = criterion_mae(ssim_map_pred, ssim_map_true)
                val_loss_mse += loss_mse.item()
                val_loss_mae += loss_mae.item()

                for k in range(len(ssim_map_true)):
                    if not os.path.exists(f'val_results/{epoch+1}'):
                        os.makedirs(f'val_results/{epoch+1}')
                    original_path = os.path.join('val_results', f'{epoch+1}/ssim_map_true_{k+s}.mat')
                    reconstructed_path = os.path.join('val_results', f'{epoch+1}/ssim_map_pred_{k+s}.mat')
                
                    original_img = np.transpose(ssim_map_true[k].cpu().numpy(), (1, 2, 0))
                    savemat(original_path, {'ssim_map_true': original_img})
                    
                    reconstructed_img = np.transpose(ssim_map_pred[k].cpu().numpy(), (1, 2, 0))
                    savemat(reconstructed_path, {'ssim_map_pred': reconstructed_img})
                s += len(ssim_map_true)
        
        val_loss_mse /= len(val_loader)
        val_loss_mae /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss (MSE): {loss_mse.item():.4f}, Validation Loss (MSE): {val_loss_mse:.4f}, Validation Loss (MAE): {val_loss_mae:.4f}')
        torch.save(model.state_dict(), os.path.join('weights', f'checkpoint_{epoch+1}.pt'))
        
    print('Training finished.')

if __name__ == "__main__":
    train_dir = 'path/to/train_dir'
    val_dir = 'path/to/val_dir'
    test_dir = 'path/to/test_dir'
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-3

    train_loader, val_loader, test_loader = load_data(train_dir, val_dir, test_dir, batch_size)

    model = None  # Replace with your model initialization
    train_model(model, train_loader, val_loader, num_epochs, learning_rate)
