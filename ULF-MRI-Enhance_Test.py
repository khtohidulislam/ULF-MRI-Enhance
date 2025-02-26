import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
from torchvision import transforms
import torch.nn.functional as F  # For interpolation
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt  # Used only for saving images via imsave

from Model_EnhancedSiren import EnhancedSiren

def load_image(image_path, slice_number, transform=None):
    """
    Loads a specific slice from the MRI image and applies preprocessing.
    
    Parameters:
        image_path (str): Path to the MRI image file.
        slice_number (int): Slice index to load.
        transform (callable, optional): Transformations to apply (e.g., resizing).
        
    Returns:
        torch.Tensor: Processed image tensor with shape [1, C, H, W].
    """
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    image_slice = img_array[slice_number, :, :]
    image = Image.fromarray(image_slice)
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def test_model_for_slice(slice_number, config, device):
    """
    Processes a single slice: loads the image, loads the corresponding trained model,
    computes enhanced images and upscaled versions, and saves the outputs along with
    a side-by-side comparison of original and enhanced upscaled images.
    
    Parameters:
        slice_number (int): The slice index to process.
        config (argparse.Namespace): Configuration parameters.
        device (torch.device): Device for computation.
    """
    print(f"Processing slice {slice_number}...")
    
    # Define transformation (resize image to 224x224)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    
    # Load the content (low-field) image for the given slice.
    content = load_image(config.content, slice_number, transform=transform).to(device)
    
    # Load the trained Enhanced SIREN model for this slice.
    model_inr = EnhancedSiren().to(device)
    model_path = os.path.join("saved_models", f"model_slice_{slice_number}.pth")
    model_inr.load_state_dict(torch.load(model_path, map_location=device))
    model_inr.eval()
    
    # Create a coordinate grid for a 224x224 image.
    res_x, res_y = 224, 224
    X = torch.linspace(-1, 1, steps=res_x)
    Y = torch.linspace(-1, 1, steps=res_y)
    x, y = torch.meshgrid(X, Y, indexing="ij")
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)
    
    # Compute enhanced image at 224x224 using the trained model.
    with torch.no_grad():
        model_output = model_inr(pixel_coordinates)
    enhanced = model_output.view(res_x, res_y).cpu().numpy()
    
    # Set upscale resolution (e.g., 896x896).
    upscale_res = 2240
    
    # Create high-resolution coordinate grid.
    X_hr = torch.linspace(-1, 1, steps=upscale_res)
    Y_hr = torch.linspace(-1, 1, steps=upscale_res)
    x_hr, y_hr = torch.meshgrid(X_hr, Y_hr, indexing="ij")
    pixel_coordinates_hr = torch.cat((x_hr.reshape(-1, 1), y_hr.reshape(-1, 1)), dim=1).to(device)
    
    # Compute INR upscaled enhanced image at 896x896.
    with torch.no_grad():
        model_output_hr = model_inr(pixel_coordinates_hr)
    enhanced_upscaled = model_output_hr.view(upscale_res, upscale_res).cpu().numpy()
    
    # Upscale the 224x224 enhanced image using conventional bicubic interpolation.
    enhanced_tensor = torch.from_numpy(enhanced).unsqueeze(0).unsqueeze(0).to(device)
    enhanced_interpolated_tensor = F.interpolate(enhanced_tensor, size=(upscale_res, upscale_res), mode='bicubic', align_corners=True)
    enhanced_interpolated = enhanced_interpolated_tensor.squeeze().cpu().numpy()
    
    # Upscale the original content image (224x224) using bicubic interpolation.
    content_interpolated_tensor = F.interpolate(content, size=(upscale_res, upscale_res), mode='bicubic', align_corners=True)
    original_upscaled = content_interpolated_tensor.cpu().squeeze().numpy()
    
    # Save individual output images.
    os.makedirs("results", exist_ok=True)
    plt.imsave(f'results/original_{slice_number}.png', content.cpu().squeeze().numpy(), cmap='gray', vmin=0, vmax=1)
    plt.imsave(f'results/original_upscaled_{slice_number}.png', original_upscaled, cmap='gray', vmin=0, vmax=1)
    plt.imsave(f'results/enhanced_{slice_number}.png', enhanced, cmap='gray', vmin=0, vmax=1)
    plt.imsave(f'results/enhanced_upscaled_{slice_number}.png', enhanced_upscaled, cmap='gray', vmin=0, vmax=1)
    plt.imsave(f'results/enhanced_interpolated_{slice_number}.png', enhanced_interpolated, cmap='gray', vmin=0, vmax=1)
    
    # Create a side-by-side comparison of original_upscaled and enhanced_upscaled.
    # Horizontally concatenate the two images.
    comparison = np.hstack((original_upscaled, enhanced_upscaled))
    plt.imsave(f'results/comparison_{slice_number}.png', comparison, cmap='gray', vmin=0, vmax=1)

def main(config):
    # Set up device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    
    # Loop over slices from 30 to 129.
    for slice_number in range(0, 30):
        test_model_for_slice(slice_number, config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Input images and field strength.
    parser.add_argument('--content', type=str, default='MRI_data/MRI_mT_sub2.nii.gz',
                        help="Path to the low-field MRI image (e.g., 64mT).")
    # Note: The style image is not used in this test code.
    parser.add_argument('--style', type=str, default='MRI_data/MRI_7T_Reg.nii.gz',
                        help="Path to the high-field (7T) style MRI image.")
    parser.add_argument('--field_strength', type=str, default="3T",
                        help="Field strength of the content image: 7T, 3T, or 1.5T.")
    
    # Test-specific parameters.
    parser.add_argument('--upscale_size', type=int, default=896,
                        help="Output resolution for upscaled INR output (e.g., 896 for 896x896).")
    
    config = parser.parse_args()
    main(config)
