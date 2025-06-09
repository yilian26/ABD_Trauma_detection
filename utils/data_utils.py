import scipy 
import torch


def resize_tensor(input_tensor, target_size):
    """
    Resize the input tensor to the target size using scipy's zoom function.
    
    Parameters:
    input_tensor (torch.Tensor): The input tensor, shape (n, c, d, h, w)
    target_size (tuple): The target size, format (n, c, d, h, w)
    
    Returns:
    torch.Tensor: Resized tensor.
    """
    # Ensure tensor is on CPU and convert to numpy
    input_numpy = input_tensor.cpu().numpy()
    
    # Calculate the zoom factors
    zoom_factors = [n / o for n, o in zip(target_size, input_numpy.shape)]
    
    # Apply zoom
    resized_numpy = scipy.ndimage.zoom(input_numpy, zoom_factors, order=1)  # Use linear interpolation
    
    # Convert back to tensor
    resized_tensor = torch.from_numpy(resized_numpy).to(input_tensor.device)
    
    return resized_tensor    


def process_mask_with_thresholding(mask, threshold):
    """
    Apply thresholding to the prediction mask to remove fragments and fill holes.
    
    Parameters:
    mask (numpy.ndarray): The prediction mask.
    threshold (float): The threshold value to apply.
    
    Returns:
    numpy.ndarray: The processed mask.
    """
    # Thresholding
    thresholded_mask = np.zeros_like(mask)
    thresholded_mask[mask > threshold] = mask[mask > threshold]

    
    # Normalization (if necessary, adjust based on the mask data range)
    # thresholded_mask = (thresholded_mask - threshold) / (thresholded_mask.max() - threshold)
#     mask = 2 * mask - 1
    
    return thresholded_mask