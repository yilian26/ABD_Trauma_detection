import scipy 
import torch
import pandas as pd
import numpy as np


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

def str2num_or_false(v):
    if v.isdigit():
        return int(v)
    else:
        return False

def convert_date(x):
    if pd.isna(x):  # Check if the value is NaN
        return x  # If it's NaN, return it as-is
    else:
        return pd.to_datetime(int(x), format="%Y%m%d")

def str2mode(x):
    if x in ["0", "cls", "classification"]:
        return "classification"
    elif x in ["1", "seg", "segmentation"]:
        return "segmentation"
    else:
        raise argparse.ArgumentTypeError("Mode must be 0/1 or classification/segmentation")

def str2dataset(x):
    if x in ["0", "cgmh"]:
        return "cgmh"
    elif x in ["1", "rsna"]:
        return "rsna"
    elif x in ["2", "multi", "multiple"]:
        return "multi"
    else:
        raise argparse.ArgumentTypeError("Dataset must be 0/1/2 or cgmh/rsna/multiple")


def prepare_labels(test_df, class_type, y_pre):
    y_pre = y_pre
    if class_type == "all":
        organ_list = ["Inj_sold"]
        y_label = test_df["inj_solid"].values
    elif class_type == "multiple":
        organ_list = ["Liver","Spleen","Kidney","Inj_sold"]
        y_label = np.stack([test_df['liver_label'].values, test_df['spleen_label'].values,  test_df['kidney_label'].values,test_df["inj_solid"].values], axis=0)
        y_pre = np.concatenate((y_pre, np.zeros((y_pre.shape[0], 1, y_pre.shape[2]))), axis=1)
        for col_idx in range(y_pre.shape[2]):  # Iterate over the third dimension
            y_pre[:, 3, col_idx] = np.maximum.reduce([y_pre[:, 0, col_idx], y_pre[:, 1, col_idx], y_pre[:, 2, col_idx]])
    else:
        organ_list = [class_type]
        y_label = test_df[f"{class_type}_label"].values

    return y_label, y_pre, organ_list

