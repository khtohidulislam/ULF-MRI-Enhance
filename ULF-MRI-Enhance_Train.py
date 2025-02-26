import os
import argparse
import torch
import numpy as np
import SimpleITK as sitk
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from models import VGGNet
from Model_EnhancedSiren import EnhancedSiren

def load_image(image_path, slice_number, field_strength, transform=None):
    """
    Loads a specific slice from an MRI image and applies preprocessing.
    
    Parameters:
        image_path (str): Path to the MRI image file.
        slice_number (int): The slice index to load.
        field_strength (str): The field strength identifier ('7T', '3T', '1.5T').
        transform (callable, optional): Transformations to apply to the image.
        
    Returns:
        torch.Tensor: The processed image tensor with an added batch dimension.
    """
    # Read image using SimpleITK and extract the specified slice.
    img_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
    image_slice = img_array[slice_number, :, :]
    
    # Apply field-specific preprocessing.
    if field_strength == "7T":
        min_val = image_slice.min()
        max_val = image_slice.max()
        scaled_image = 255 * (image_slice - min_val) / (max_val - min_val)
        # Crop if needed (as per previous code).
        uint8_image = scaled_image.astype(np.uint8)[12:304-36, :]
        image = Image.fromarray(uint8_image)
    else:
        image = Image.fromarray(image_slice)
    
    # Apply transformations if provided.
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def train_model_for_slice(slice_number, config, device):
    """
    Trains the Enhanced SIREN model for a given slice and saves the trained model.
    
    Parameters:
        slice_number (int): The MRI slice number to train on.
        config (argparse.Namespace): Configuration parameters.
        device (torch.device): Device to perform training on.
    """
    print(f"\nTraining for slice {slice_number}...")
    
    # Define transformation pipeline (resize to 224x224).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    
    # Load the low-field (content) image and high-field (style) image for the given slice.
    content = load_image(config.content, slice_number, config.field_strength, transform=transform).to(device)
    style = load_image(config.style, slice_number, "7T", transform=transform).to(device)
    
    # Initialize the Enhanced SIREN and VGG models.
    model_inr = EnhancedSiren().to(device)
    model_vgg = VGGNet().to(device).eval()  # VGG is used only for feature extraction.
    
    # Set up the optimizer.
    optimizer = torch.optim.Adam(model_inr.parameters(), lr=config.lr)
    
    # Create a coordinate grid for INR input (for a 224x224 image).
    res_x, res_y = 224, 224
    X = torch.linspace(-1, 1, steps=res_x)
    Y = torch.linspace(-1, 1, steps=res_y)
    x, y = torch.meshgrid(X, Y, indexing="ij")
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)
    
    # Training loop (no visualization to speed up training).
    for iter in tqdm(range(config.total_iterations), desc=f"Slice {slice_number}"):
        # Forward pass through the INR model.
        model_output = model_inr(pixel_coordinates)
        target = model_output.view(res_x, res_y).unsqueeze(0).unsqueeze(0)
        
        # Repeat channels to match the 3-channel input expected by VGG.
        target_in = target.repeat(1, 3, 1, 1)
        content_in = content.repeat(1, 3, 1, 1)
        style_in = style.repeat(1, 3, 1, 1)
        
        # Extract features from VGG.
        all_target_features = model_vgg(target_in)
        all_content_features = model_vgg(content_in)
        all_style_features = model_vgg(style_in)
        
        # If specific VGG layers are provided, filter the features.
        if config.vgg_layers:
            layer_indices = [int(idx) for idx in config.vgg_layers.split(',')]
            target_features = [all_target_features[i] for i in layer_indices]
            content_features = [all_content_features[i] for i in layer_indices]
            style_features = [all_style_features[i] for i in layer_indices]
        else:
            target_features = all_target_features
            content_features = all_content_features
            style_features = all_style_features
        
        # Compute the reconstruction loss (L1 loss).
        recon_loss = torch.mean(torch.abs(target - content))
        
        # Compute content and style losses based on VGG features.
        content_loss = 0
        style_loss = 0
        for f_t, f_c, f_s in zip(target_features, content_features, style_features):
            content_loss += torch.mean((f_t - f_c) ** 2)
            # Compute Gram matrices for style loss.
            _, c, h, w = f_t.size()
            f_t_reshaped = f_t.view(c, h * w)
            f_s_reshaped = f_s.view(c, h * w)
            gram_t = torch.mm(f_t_reshaped, f_t_reshaped.t())
            gram_s = torch.mm(f_s_reshaped, f_s_reshaped.t())
            style_loss += torch.mean((gram_t - gram_s) ** 2) / (c * h * w)
        
        # Total loss: weighted sum of content, style, and reconstruction losses.
        total_loss = config.a1 * content_loss + config.a2 * style_loss + config.a3 * recon_loss
        
        # Backpropagation and optimization.
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Save the final model weights for this slice.
    model_save_path = os.path.join("saved_models", f"model_slice_{slice_number}.pth")
    torch.save(model_inr.state_dict(), model_save_path)
    print(f"Model for slice {slice_number} saved to {model_save_path}")

def main(config):
    # Set up device.
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create necessary directories if they do not exist.
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)  # Retained if needed later.

    # Loop over the specified slice range and train the model for each slice.
    for slice_num in range(config.start_slice, config.end_slice + 1):
        train_model_for_slice(slice_num, config, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Input images and field strength.
    parser.add_argument('--content', type=str, default='MRI_data/MRI_mT_sub2.nii.gz',
                        help="Path to the low-field MRI image (e.g., 64mT).")
    parser.add_argument('--style', type=str, default='MRI_data/MRI_7T_Reg.nii.gz',
                        help="Path to the high-field (7T) style MRI image.")
    parser.add_argument('--field_strength', type=str, default="3T",
                        help="Field strength of the content image: 7T, 3T, or 1.5T.")
    
    # Training hyperparameters.
    parser.add_argument('--total_iterations', type=int, default=1001,
                        help="Number of training iterations for each slice.")
    parser.add_argument('--a1', type=float, default=3,
                        help="Content loss weight.")
    parser.add_argument('--a2', type=float, default=15,
                        help="Style loss weight.")
    parser.add_argument('--a3', type=float, default=1,
                        help="Reconstruction loss weight.")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU ID (if applicable).")
    
    # Optional: Specify VGG layers to use for loss computation.
    parser.add_argument('--vgg_layers', type=str, default="0,1,2,3,4",
                        help="Comma-separated list of VGG layer indices to use (e.g., '0,1,2,3,4').")
    
    # Slice range for training (e.g., middle 100 slices).
    parser.add_argument('--start_slice', type=int, default=130,
                        help="Starting slice index for training.")
    parser.add_argument('--end_slice', type=int, default=159,
                        help="Ending slice index for training.")
    
    # Upscaling parameter for final output (if needed).
    parser.add_argument('--upscale_size', type=int, default=448,
                        help="Output resolution for upscaled INR output (e.g., 448 for 448x448).")
    
    config = parser.parse_args()
    main(config)
