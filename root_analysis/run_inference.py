"""Inference module for segmentation model predictions.

This module provides functions to run inference on images using trained
segmentation models. It handles patch-based prediction with overlap,
ROI detection, and reconstruction of full-resolution masks.
"""

import platform
from pathlib import Path
from typing import Union, List, Tuple, Dict

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import keras.backend as K
from patchify import patchify

from library.roi import detect_roi, crop_to_roi
from library.patch_dataset import padder, restore_mask_to_original


def f1(y_true, y_pred):
    """Calculate F1 score metric for binary segmentation.
    
    Computes F1 score as the harmonic mean of precision and recall.
    
    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted probabilities or binary labels.
    
    Returns:
        F1 score as a scalar tensor.
    """
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def configure_tensorflow_for_platform():
    """Configure TensorFlow based on the current platform.
    
    Detects if running on Mac with Metal support, Linux/Windows with CUDA,
    or CPU-only. Configures TensorFlow accordingly and enables memory growth
    for GPU devices.
    
    Returns:
        str: Device type - 'metal', 'cuda', or 'cpu'.
    """
    system = platform.system()
    
    if system == 'Darwin':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                return 'metal'
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                return 'cpu'
        return 'cpu'
    else:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            return 'cuda'
        return 'cpu'


def load_segmentation_model(model_path, verbose=True):
    """Load a Keras segmentation model with custom F1 metric.
    
    Automatically detects platform and configures TensorFlow before loading
    the model.
    
    Args:
        model_path: Path to the saved model file (.h5 or .keras).
        verbose: If True, print device information.
    
    Returns:
        Loaded Keras model ready for inference.
    """
    device = configure_tensorflow_for_platform()
    
    if verbose:
        print(f"Using device: {device}")
    
    model = load_model(model_path, custom_objects={"f1": f1}, compile=False)
    
    return model


def predict_patches_batched(model, patches, patch_size, batch_size=8, verbose=True):
    """Predict on all patches using batched processing.
    
    Processes patches in batches to manage memory usage effectively.
    
    Args:
        model: Loaded Keras model.
        patches: Patchified image array with shape (n_rows, n_cols, 1, H, W, C).
        patch_size: Size of each square patch.
        batch_size: Number of patches to process at once.
        verbose: If True, print progress information.
    
    Returns:
        Array of predicted patches with shape (n_rows, n_cols, 1, H, W, 1),
        with values as uint8 (0 or 255).
    """
    n_rows, n_cols = patches.shape[0], patches.shape[1]
    
    predicted_patches = np.zeros((n_rows, n_cols, 1, patch_size, patch_size, 1), dtype=np.uint8)
    
    all_patches = []
    patch_positions = []
    
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            patch = patches[row_idx, col_idx, 0]
            patch_normalized = patch.astype(np.float32) / 255.0
            all_patches.append(patch_normalized)
            patch_positions.append((row_idx, col_idx))
    
    all_patches = np.array(all_patches)
    total_patches = len(all_patches)
    
    if verbose:
        print(f"Total patches: {total_patches}")
    
    num_batches = (total_patches + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Processing in {num_batches} batches of size {batch_size}")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_patches)
        
        batch_patches = all_patches[start_idx:end_idx]
        batch_predictions = model.predict(batch_patches, verbose=0)
        
        for i, pred in enumerate(batch_predictions):
            patch_idx = start_idx + i
            row_idx, col_idx = patch_positions[patch_idx]
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            predicted_patches[row_idx, col_idx, 0] = pred_binary
        
        if (batch_idx + 1) % 5 == 0:
            tf.keras.backend.clear_session()
        
        if verbose:
            print(f"Batch {batch_idx + 1}/{num_batches} ({end_idx}/{total_patches} patches)", end='\r')
    
    if verbose:
        print()
    
    return predicted_patches


def prepare_image_for_prediction(image, patch_size, step_size):
    """Prepare an image for patch-based prediction.
    
    Performs preprocessing including ROI detection, cropping, padding,
    and patch extraction.
    
    Args:
        image: Input grayscale image as numpy array with shape (H, W).
        patch_size: Size of square patches.
        step_size: Step size for patch extraction (controls overlap).
    
    Returns:
        Dictionary containing:
            - 'patches': Patchified array with shape (n_rows, n_cols, 1, H, W, C)
            - 'roi_box': ROI coordinates as ((x1, y1), (x2, y2))
            - 'padded_shape': Shape of padded image (H, W, C)
            - 'original_shape': Shape of original input image (H, W)
    """
    original_shape = image.shape
    
    roi_box = detect_roi(image)
    cropped_image = crop_to_roi(image, roi_box)
    
    padded_image = padder(cropped_image, patch_size, step_size)
    
    if len(padded_image.shape) == 2:
        padded_image = np.expand_dims(padded_image, axis=-1)
    
    padded_shape = padded_image.shape
    
    patches = patchify(padded_image, (patch_size, patch_size, 1), step=step_size)
    
    return {
        'patches': patches,
        'roi_box': roi_box,
        'padded_shape': padded_shape,
        'original_shape': original_shape
    }


def unpatchify_with_overlap(patches, target_shape, patch_size, step_size):
    """Reconstruct full image from overlapping patches.
    
    Uses averaging for overlapping regions to create smooth transitions.
    
    Args:
        patches: Predicted patches with shape (n_rows, n_cols, 1, H, W, 1).
        target_shape: Target shape for reconstructed image (H, W, C).
        patch_size: Size of square patches.
        step_size: Step size used during patch extraction.
    
    Returns:
        Reconstructed image as numpy array with shape (H, W).
    """
    n_rows, n_cols = patches.shape[0], patches.shape[1]
    
    reconstructed = np.zeros((target_shape[0], target_shape[1]), dtype=np.float32)
    count_map = np.zeros((target_shape[0], target_shape[1]), dtype=np.float32)
    
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            patch = patches[row_idx, col_idx, 0, :, :, 0]
            
            start_row = row_idx * step_size
            start_col = col_idx * step_size
            end_row = min(start_row + patch_size, target_shape[0])
            end_col = min(start_col + patch_size, target_shape[1])
            
            patch_height = end_row - start_row
            patch_width = end_col - start_col
            
            reconstructed[start_row:end_row, start_col:end_col] += patch[:patch_height, :patch_width]
            count_map[start_row:end_row, start_col:end_col] += 1
    
    count_map[count_map == 0] = 1
    reconstructed = reconstructed / count_map
    
    return reconstructed.astype(np.uint8)


def predict_single_image(model, image, patch_size, step_size, batch_size=8, verbose=True):
    """Run complete inference pipeline on a single image.
    
    Args:
        model: Loaded Keras model.
        image: Input grayscale image as numpy array with shape (H, W).
        patch_size: Size of square patches for prediction.
        step_size: Step size for patch extraction (controls overlap).
        batch_size: Number of patches to process at once.
        verbose: If True, print progress information.
    
    Returns:
        Binary mask as numpy array with shape (H, W) matching input image,
        with values 0 (background) and 255 (foreground).
    """
    if verbose:
        print(f"Processing image with shape {image.shape}")
    
    result = prepare_image_for_prediction(image, patch_size, step_size)
    if verbose:
        print(f"Created {result['patches'].shape[0]}x{result['patches'].shape[1]} patch grid")
    
    predicted_patches = predict_patches_batched(
        model, result['patches'], patch_size, batch_size, verbose
    )
    
    if verbose:
        print("Reconstructing image from patches...")
    reconstructed = unpatchify_with_overlap(
        predicted_patches, result['padded_shape'], patch_size, step_size
    )
    
    full_mask = restore_mask_to_original(
        reconstructed, result['original_shape'], result['roi_box']
    )
    
    if verbose:
        print(f"Final mask shape: {full_mask.shape}")
    
    return full_mask


def run_inference(
    model_path: Union[str, Path],
    image_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    mask_type: str,
    patch_size: int = 256,
    step_size: int = 128,
    batch_size: int = 8,
    verbose: bool = True
) -> int:
    """Run inference on multiple images using a trained segmentation model.
    
    This is the main convenience function that handles model loading, image
    processing, and saving predictions. Output files are saved to a directory
    named after the model stem.
    
    Args:
        model_path: Path to the trained model file (.h5 or .keras).
        image_paths: List of paths to input images (as Path objects or strings).
        output_dir: Base directory where outputs will be saved.
        mask_type: Type of mask being predicted (e.g., 'root', 'shoot').
        patch_size: Size of square patches for prediction. Default is 256.
        step_size: Step size for patch extraction. Default is 128.
        batch_size: Number of patches to process at once. Default is 8.
        verbose: If True, print progress information. Default is True.
    
    Returns:
        Number of images successfully processed.
    
    Output structure:
        output_dir/
            model_name/
                image1_mask_type.png
                image2_mask_type.png
    
    Example:
        >>> image_list = ['img1.png', 'img2.png', 'img3.png']
        >>> n = run_inference(
        ...     model_path='shoots.h5',
        ...     image_paths=image_list,
        ...     output_dir='./predictions',
        ...     mask_type='shoot',
        ...     patch_size=256,
        ...     step_size=128
        ... )
        >>> print(f"Processed {n} images")
    """
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    image_paths = [Path(p) for p in image_paths]
    
    model_name = model_path.stem
    
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Loading model from {model_path}")
    model = load_segmentation_model(model_path, verbose=verbose)
    
    if verbose:
        print(f"Processing {len(image_paths)} images")
        print(f"Output directory: {model_output_dir}")
    
    processed_count = 0
    
    for i, img_path in enumerate(image_paths, 1):
        if not img_path.exists():
            print(f"Warning: {img_path} does not exist, skipping")
            continue
        
        if verbose:
            print(f"\n[{i}/{len(image_paths)}] Processing {img_path.name}")
        
        image = cv2.imread(str(img_path), 0)
        
        if image is None:
            print(f"Warning: Failed to load {img_path}, skipping")
            continue
        
        mask = predict_single_image(model, image, patch_size, step_size, batch_size, verbose)
        
        output_filename = f"{img_path.stem}_{mask_type}.png"
        output_path = model_output_dir / output_filename
        
        cv2.imwrite(str(output_path), mask)
        
        if verbose:
            print(f"Saved mask to {output_path}")
        
        processed_count += 1
    
    if verbose:
        print(f"\nCompleted processing {processed_count}/{len(image_paths)} images")
    
    return processed_count, model_output_dir