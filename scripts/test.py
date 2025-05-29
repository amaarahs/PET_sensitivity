import os
import sys
import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import sirf.STIR as stir


dir_path = os.path.dirname(os.getcwd())
source_path = os.path.join(dir_path, 'source')
sys.path.append(source_path)
output_dir = os.path.join(dir_path, 'data', 'processed')
model_path = os.path.join(dir_path, 'data', 'trained_models', 'grid_VanillaCNN_lr0.01_bs32_e20', 'final_model.pth')

sys.path.append(source_path)
sys.path.append(os.path.join(source_path, 'models'))
dir_path = os.path.dirname(os.getcwd())

from models.UNet import UNet
from reconstruction.registration import generate_transformed_image

class BrainWebTestDataset(Dataset):
    def __init__(self, sens_images, attn_images):
        self.sens_images = sens_images
        self.attn_images = attn_images

    def __len__(self):
        return len(self.sens_images)

    def __getitem__(self, idx):
        sens = self.sens_images[idx]
        attn = self.attn_images[idx]

        transformed_sens, transformed_attn = generate_transformed_image(sens, attn)

        x_sens = sens.as_array().squeeze()
        x_attn = transformed_attn.as_array().squeeze()
        x = np.array([x_sens, x_attn])  # shape: [2, H, W]

        y = transformed_sens.as_array().squeeze(0)  # shape: [H, W]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=2, n_class=1).to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")


sens_images = [stir.ImageData(os.path.join(output_dir, f'sens_{i}.hv')) for i in range(5)]
attn_images = [stir.ImageData(os.path.join(output_dir, f'attn_{i}.hv')) for i in range(5)]

test_dataset = BrainWebTestDataset(sens_images, attn_images)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model.eval()
with torch.no_grad():
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        out = model(X)

        vmax = max([X[0, 0].max(), y[0].max()])

        plt.figure(figsize=(8, 2))
        plt.subplot(1, 3, 1)
        plt.imshow(X[0, 0].cpu().numpy(), vmax=vmax)
        plt.title('Original Sensitivity')

        plt.subplot(1, 3, 2)
        plt.imshow(out[0, 0].cpu().numpy(), vmax=vmax)
        plt.title('Model Output')

        plt.subplot(1, 3, 3)
        plt.imshow(y[0].cpu().numpy(), vmax=vmax)
        plt.title('Ground Truth')

        plt.xticks([]); plt.yticks([])
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f'output_image_{i}.png')  # save instead of showing
        plt.close()
