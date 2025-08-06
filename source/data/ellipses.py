### Ellipse trainind dataset for PyTorch ###
### Creates sensitivity images and corresponding CT images ###
### Sam Porter 1st verstion 2023-29-11 ###

import numpy as np
import torch
from sirf.STIR import AcquisitionSensitivityModel, AcquisitionModelUsingRayTracingMatrix
from .misc import random_phantom, affine_transform_2D_image

MAX_PHANTOM_INTENSITY = 0.096 * 2
mu_water = 0.096

def generate_random_transform_values() -> tuple:
    """
    Generates random values for affine transformation.
    """
    theta = np.random.uniform(-np.pi/16, np.pi/16)
    tx, ty = np.random.uniform(-1, 1), np.random.uniform(-1, 5)
    sx, sy = np.random.uniform(0.95, 1.05), np.random.uniform(0.95, 1.05)
    return theta, tx, ty, sx, sy

def make_max_n(image: np.ndarray, n: float) -> np.ndarray:
    """
    Scales the image so that its maximum is n.
    """
    image *= n / image.max()
    return image

class EllipsesDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for simulated ellipses.

    Parameters:
        radon_transform: SIRF acquisition model used as the forward operator.
        attenuation_image_template: SIRF image data for the template.
        sinogram_template: Template for sinogram data.
        attenuation: Boolean flag for attenuation.
        no_att_sens: Boolean flag for sensitivity without attenuation.
        num_samples: Number of samples in the dataset.
        mode: Dataset mode (train, validation, test).
        seed: Random seed for phantom generation.
        norm_mode: Method of normalisation used
            - 'max': Normalize by the max of non-attenuated sensitivity.
            - 'mean': Normalize by the mean of non-attenuated sensitivity.
            - 'none': No normalisation applied to sensitivity images.
    """

    def __init__(self, attenuation_image_template, sinogram_template, num_samples,
                 generate_non_attenuated_sensitivity=False, norm_mode='max'):
                 
        self.num_samples = num_samples

        self.radon_transform = AcquisitionModelUsingRayTracingMatrix()
        self.radon_transform.set_up(sinogram_template, attenuation_image_template)

        self.tmp_acq_model = AcquisitionModelUsingRayTracingMatrix()

        self.attenuation_image_template = attenuation_image_template.clone()

        self.template = sinogram_template
        self.one_sino = sinogram_template.get_uniform_copy(1)

        self.generate_non_attenuated_sensitivity = generate_non_attenuated_sensitivity
        
        self.norm_mode = norm_mode

    def _get_sensitivity_image(self, ct_image, attenuation=True):
        """
        Generates the sensitivity image.
        """        
        if attenuation:
            asm_attn = AcquisitionSensitivityModel(ct_image, self.radon_transform)
            asm_attn.set_up(self.template)
            self.tmp_acq_model.set_acquisition_sensitivity(asm_attn)
        else:
            asm_attn = AcquisitionSensitivityModel(ct_image.get_uniform_copy(0), self.radon_transform)
            asm_attn.set_up(self.template)
            self.tmp_acq_model.set_acquisition_sensitivity(asm_attn)
            
        self.tmp_acq_model.set_up(self.template, ct_image)
        
        return self.tmp_acq_model.backward(self.one_sino)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        random_phantom_array = make_max_n(random_phantom(self.attenuation_image_template.shape, 20), MAX_PHANTOM_INTENSITY)
        ct_image = self.attenuation_image_template.clone().fill(random_phantom_array)
        sens_image = self._get_sensitivity_image(ct_image)
        
        theta, tx, ty, sx, sy = generate_random_transform_values()
        ct_image_transform = affine_transform_2D_image(theta, tx, ty, sx, sy, ct_image)
        ct_image_transform.move_to_scanner_centre(self.template)
        sens_image_transform = self._get_sensitivity_image(ct_image_transform)
        sens_image_no_att = self._get_sensitivity_image(ct_image, attenuation=False)
        
        # normalise images by either the max or mean of non-attenuated sensitivity image 
        if self.norm_mode == 'max':
            denom = sens_image_no_att.as_array().max() + 1e-8
        elif self.norm_mode == 'mean':
            denom = np.mean(sens_image_no_att.as_array()) + 1e-8
        elif self.norm_mode == 'none':
            denom = 1
        else:
            raise ValueError(f"Invalid normalisation mode: {self.norm_mode}")
            
        norm_sens_image = sens_image.as_array()/denom
        norm_sens_image_transform = sens_image_transform.as_array()/denom
        norm_sens_image_no_att = sens_image_no_att.as_array()/denom

        
        if self.norm_mode =='none':
            norm_ct_image = ct_image.as_array()
            norm_ct_image_transform = ct_image_transform.as_array()
        else:
            norm_ct_image = ct_image.as_array()/mu_water
            norm_ct_image_transform = ct_image_transform.as_array()/mu_water
        
        # includes non-attenuated sensitivity image in inputs
        if self.generate_non_attenuated_sensitivity:
            return np.array([norm_sens_image.squeeze(), norm_sens_image_no_att.squeeze(), norm_ct_image.squeeze(), norm_ct_image_transform.squeeze()]), norm_sens_image_transform.squeeze() 
        
        # returns normalised original sensitivity , original attenuation, transformed attenuation images for inputs and transformed senstivity for target
        return np.array([norm_sens_image.squeeze(), norm_ct_image.squeeze(), norm_ct_image_transform.squeeze()]), norm_sens_image_transform.squeeze()
 
