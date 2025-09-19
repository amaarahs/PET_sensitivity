from torch.utils.data import Dataset
import torch
import numpy as np
import sirf.STIR as stir


mu_water = 0.096

class BrainWebTestDataset(Dataset):
    def __init__(self, sens_images, attn_images, emis_images, transformed_attn_images, transformed_sens_images, template_sinogram, generate_non_attenuated_sensitivity=False, norm_mode='max'): 
        self.sens_images = sens_images
        self.attn_images = attn_images
        self.emis_images = emis_images
        self.transformed_attn_images = transformed_attn_images
        self.transformed_sens_images = transformed_sens_images
        self.template_sinogram = template_sinogram
        self.generate_non_attenuated_sensitivity = generate_non_attenuated_sensitivity
        self.norm_mode = norm_mode

    def __len__(self):
        return len(self.sens_images)

    def __getitem__(self, idx):
        sens = self.sens_images[idx]
        attn = self.attn_images[idx]
        emis = self.emis_images[idx]
        transformed_attn = self.transformed_attn_images[idx]
        transformed_sens = self.transformed_sens_images[idx]
        
        # Get data as arrays
        sens_array = sens.as_array()
        attn_array = attn.as_array()
        trans_attn_array = transformed_attn.as_array()
        trans_sens_array = transformed_sens.as_array()

        
        # Create non-attenuated sensitivity images for normalisation
        acq_model_no_attn = stir.AcquisitionModelUsingRayTracingMatrix()
        acq_model_no_attn.set_num_tangential_LORs(10)
        acq_model_no_attn.set_up(self.template_sinogram, emis)
        uniform_acq_data = self.template_sinogram.get_uniform_copy(1.0)
        non_att = acq_model_no_attn.backward(uniform_acq_data)
        sens_image_no_att = non_att.as_array()
        
        # normalise images by either the max or mean of non-attenuated sensitivity image 
        if self.norm_mode == 'max':
            denom = sens_image_no_att.max() + 1e-8
        elif self.norm_mode == 'mean':
            denom = np.mean(sens_image_no_att) + 1e-8
        elif self.norm_mode == 'median':
            denom = np.median(sens_image_no_att) + 1e-8
        elif self.norm_mode == 'none':
            denom = 1
        else:
            raise ValueError(f"Invalid normalisation mode: {self.norm_mode}")

        # Normalise data
        sens_norm = sens_array/denom
        trans_sens_norm = trans_sens_array/denom
        sens_ratio = (trans_sens_norm / (sens_norm + 1e-8)) - 1

        
        # Normalise attenuation images (unless norm_mode is none)
        if self.norm_mode == 'none':
            attn_norm = attn_array
            trans_attn_norm = trans_attn_array
        else:
            attn_norm = attn_array / mu_water
            trans_attn_norm = trans_attn_array / mu_water
            
        attn_diff = trans_attn_norm - attn_norm

        x_sens = sens_norm.squeeze()
        x_attn_diff = attn_diff.squeeze()
        x_attn_trans = trans_attn_norm.squeeze()
        
        if self.generate_non_attenuated_sensitivity:
            x_non_attn = (sens_image_no_att/denom).squeeze()
            x = np.array([x_sens, x_non_attn, x_attn_trans, x_attn_diff]) # inputs 
            
        else:
            x = np.array([x_sens, x_attn_trans, x_attn_diff]) # inputs 
        
        y = sens_ratio # target    
    

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(0)


    

