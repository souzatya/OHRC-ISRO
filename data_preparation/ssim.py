import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# Function to compute Mean Squared Error (MSE)
def compute_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

# Load two images in grayscale
image1 = cv2.imread('../data/images/calibrated/20190906/ch2_ohr_ncp_20190906T1246532096_b_brw_d18.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('../data/images/raw/20190906/ch2_ohr_nrp_20190906T1246532096_b_brw_d18.png', cv2.IMREAD_GRAYSCALE)

# Compute SSIM index and SSIM map
ssim_index, ssim_map = ssim(image1, image2, full=True)

# Compute MSE
mse_value = compute_mse(image1, image2)

# Display the SSIM index
print(f"SSIM Index: {ssim_index}")

# Display the MSE
print(f"MSE: {mse_value}")

# Display the SSIM map
plt.figure(figsize=(10, 8))
plt.imshow(ssim_map, cmap='gray')
plt.title('SSIM Map')
plt.colorbar()
plt.show()
