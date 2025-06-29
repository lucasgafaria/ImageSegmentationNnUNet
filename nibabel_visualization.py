import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import nibabel as nib

img = nib.load('C:/Users/Lucas Gafaria/Downloads/Descompactados/content/nnUNet_raw_data_base/nnUNet_raw_data/Dataset505_BraTS2020_subset/imagesTr/BraTS20_Training_117_0000.nii').get_fdata()
seg = nib.load('C:/Users/Lucas Gafaria/Downloads/Descompactados/content/nnUNet_raw_data_base/nnUNet_raw_data/Dataset505_BraTS2020_subset/predictions/BraTS20_Training_117.nii').get_fdata()

slice_idx = img.shape[2] // 2

fig, ax = plt.subplots()
img_display = ax.imshow(img[:, :, slice_idx], cmap='gray')
seg_display = ax.imshow(seg[:, :, slice_idx], cmap='nipy_spectral', alpha=0.5)
plt.axis('off')

def on_scroll(event):
    global slice_idx
    if event.button == 'up':
        slice_idx = (slice_idx + 1) % img.shape[2]
    elif event.button == 'down':
        slice_idx = (slice_idx - 1) % img.shape[2]
    
    img_display.set_data(img[:, :, slice_idx])
    seg_display.set_data(seg[:, :, slice_idx])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)
plt.show()