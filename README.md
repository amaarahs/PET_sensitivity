# Using Deep Learning to Estimate Average PET Sensitivity to Correct for Motion in PET 

This project investigates the use of deep learning to estimate motion-affected sensitivity images in Positron Emission Tomography (PET). A U-Net convolutional neural network was implemented to predict transformed sensitivity images, which were then used for image reconstruction with the Synergistic Image Reconstruction Framework (SIRF).  

Different output variations and normalisation strategies were compared, with models fine-tuned and tested on the BrainWeb dataset. Results showed that deep learning can produce sensitivity images and reconstructions close to aligned references, demonstrating the feasibility of this approach for PET motion correction.  

## Usage
- Predict motion-transformed sensitivity images using U-Net.
- Generate ellipses training data with different normalisation methods.
- Generate ellipses training data with specific inputs and outputs: transformed sensitivity image, difference between transformed and original sensitivity, and ratio between transformed and original sensitvity minus 1.
- Script to train models with this data and calculate the MSEs.
- Generate BrainWeb data and slices of the BrainWeb data.
- Test models using ellipses and BrainWeb data.
- Create BrainWeb data for fine-tuning.
- Fine-tune on the BrainWeb dataset.
- Reconstruct PET images using predicted sensitivity maps with SIRF.
- Quantitative evaluation with RMSE and nRMSE.

## Requirements
- Python 3.9+
- PyTorch (https://docs.pytorch.org/docs/stable/index.html)
- SIRF (https://github.com/SyneRBI/SIRF/wiki/Software-Documentation)
- NumPy, Matplotlib, SciPy etc.  
(See `requirements.txt` for full details.)

## Installation
```bash
# Clone the repository
git clone https://github.com/amaarahs/PET_sensitivity/
cd PET_sensitivity

# Install dependencies
pip install -r requirements.txt
