import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import imageio
import os

# Load data
img_path = "C:/Users/Lucas Gafaria/Downloads/Descompactados/content/nnUNet_raw_data_base/nnUNet_raw_data/Dataset505_BraTS2020_subset/imagesTr/BraTS20_Training_117_0000.nii"
seg_path = "C:/Users/Lucas Gafaria/Downloads/Descompactados/content/nnUNet_raw_data_base/nnUNet_raw_data/Dataset505_BraTS2020_subset/predictions/BraTS20_Training_117.nii"

img = nib.load(img_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

output_dir = "gif_frames"
os.makedirs(output_dir, exist_ok=True)

# Save overlayed slices
frames = []
for i in range(0, img.shape[2], 2):  # Use every 2nd slice to reduce size
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img[:, :, i], cmap='gray')
    ax.imshow(seg[:, :, i], cmap='nipy_spectral', alpha=0.5)
    ax.axis('off')
    frame_path = f"{output_dir}/frame_{i:03d}.png"
    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    frames.append(imageio.imread(frame_path))

# Create GIF
gif_path = "segmentation_scroll.gif"
imageio.mimsave(gif_path, frames, duration=0.1)

print(f"✅ GIF saved at: {gif_path}")
