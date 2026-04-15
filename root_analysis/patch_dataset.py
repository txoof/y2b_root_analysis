# library/patch_dataset.py
"""Functions for creating and managing patch-based datasets from images and masks."""

import os
import shutil
import json
import glob
from datetime import datetime, timezone
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from patchify import patchify, unpatchify
from tqdm import tqdm
from library.roi import detect_roi, crop_to_roi

METADATA_VERSION = "1.0"


def padder(image, patch_size, is_mask=False):
    """Add padding to an image to make its dimensions divisible by patch size.

    Calculates padding needed for both height and width so that dimensions become 
    divisible by the given patch size. Padding is applied evenly to both sides of 
    each dimension. If padding amount is odd, one extra pixel is added to the 
    bottom or right side.

    Parameters:
        image: Input image as numpy array with shape (height, width, channels).
        patch_size: The patch size to which image dimensions should be divisible.
        is_mask: If True, uses grayscale padding. If False, uses RGB padding.

    Returns:
        Padded image as numpy array with dimensions divisible by patch_size.

    Example:
        >>> padded_image = padder(cv2.imread('example.jpg'), 128)
    """
    h, w = image.shape[:2]
    
    # Calculate padding only if needed
    height_padding = 0 if h % patch_size == 0 else ((h // patch_size) + 1) * patch_size - h
    width_padding = 0 if w % patch_size == 0 else ((w // patch_size) + 1) * patch_size - w
    
    # Early return if no padding needed
    if height_padding == 0 and width_padding == 0:
        return image
    
    # Split padding evenly
    top_padding = height_padding // 2
    bottom_padding = height_padding - top_padding
    left_padding = width_padding // 2
    right_padding = width_padding - left_padding
    
    # Use black padding
    pad_value = 0
    
    return cv2.copyMakeBorder(
        image, 
        top_padding, bottom_padding, 
        left_padding, right_padding, 
        cv2.BORDER_CONSTANT, 
        value=pad_value
    )

def unpadder(padded_image, roi_box):
    """Remove padding from an image to restore original ROI dimensions.

    Calculates and removes padding that was added to make an image divisible by
    a patch size. Padding is removed evenly from both sides of each dimension.
    This function reverses the padding operation applied by the padder function.

    Parameters:
        padded_image: Padded image as numpy array with shape (height, width, channels).
        roi_box: Tuple of ((x1, y1), (x2, y2)) representing the original ROI coordinates.

    Returns:
        Cropped image as numpy array with original ROI dimensions.

    Example:
        >>> roi_box = ((776, 70), (3519, 2813))
        >>> original = unpadder(padded_image, roi_box)
    """
    h, w = padded_image.shape[:2]

    # Calculate original ROI dimensions
    roi_height = roi_box[1][1] - roi_box[0][1]
    roi_width = roi_box[1][0] - roi_box[0][0]

    # Calculate total padding in each dimension
    total_height_px = h - roi_height
    total_width_px = w - roi_width

    # Early return if no padding to remove
    if total_height_px == 0 and total_width_px == 0:
        return padded_image

    # Split padding evenly
    left_padding = total_width_px // 2
    right_padding = total_width_px - left_padding
    top_padding = total_height_px // 2
    bottom_padding = total_height_px - top_padding

    # Crop the image & handle masks too
    if padded_image.ndim == 2:
        return padded_image[top_padding:-bottom_padding, left_padding:-right_padding]
    else:
        return padded_image[top_padding:-bottom_padding, left_padding:-right_padding, :]


def restore_mask_to_original(padded_mask, original_image_shape, roi_box):
    """Restore a padded mask to match the original image dimensions.
    
    This function removes padding from a mask and places it at the correct position
    in a full-size mask that matches the original image dimensions.
    
    Parameters:
        padded_mask: Padded binary mask as numpy array with shape (height, width).
        original_image_shape: Tuple of (height, width) or (height, width, channels) 
                            from the original image.
        roi_box: Tuple of ((x1, y1), (x2, y2)) representing the original ROI coordinates.
    
    Returns:
        Binary mask as numpy array with shape matching original_image_shape[:2],
        with mask values of 0 and 255.
    
    Example:
        >>> full_mask = restore_mask_to_original(predicted_mask, image.shape, roi_box)
        >>> cv2.imwrite('output.png', full_mask)
    """
    # Remove padding using unpadder
    unpadded_mask = unpadder(padded_mask, roi_box)
    
    # Ensure mask is binary (0 and 255)
    binary_mask = (unpadded_mask > 0).astype(np.uint8) * 255
    
    # Create full-size mask matching original image dimensions
    full_mask = np.zeros(original_image_shape[:2], dtype=np.uint8)
    
    # Extract ROI coordinates
    (x1, y1), (x2, y2) = roi_box
    
    # Place the mask at the correct position
    full_mask[y1:y2, x1:x2] = binary_mask
    
    return full_mask

def apply_preprocessing_pipeline(image, preprocess_fns):
    """Apply a list of preprocessing functions sequentially to an image.
    
    Args:
        image: Numpy array with shape (H, W, C) or (H, W).
        preprocess_fns: List of callables, each taking image and returning 
            processed image. Can be None or empty list.
    
    Returns:
        Preprocessed image.
    """
    if preprocess_fns is None or len(preprocess_fns) == 0:
        return image
    
    processed = image.copy()
    for fn in preprocess_fns:
        processed = fn(processed)
    
    return processed

def process_image(image, patch_size, scaling_factor, is_mask=False):
    """Pad and scale a single image.
    
    Args:
        image: Numpy array with shape (H, W, C) or (H, W).
        patch_size: Target patch size.
        scaling_factor: Scaling factor (<=1.0).
        is_mask: Whether this is a mask (affects padding value).
    
    Returns:
        Processed image.
    """
    # Scale if needed
    if scaling_factor != 1.0:
        image = cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)
    
    # Pad to be divisible by patch_size
    image = padder(image, patch_size, is_mask=is_mask)
    
    return image


def create_patch_directories(output_dir, dataset_type, mask_types=['root', 'shoot', 'seed']):
    """Create directory structure needed for patch datasets.
    
    Args:
        output_dir: Base output directory (e.g., 'data_patched').
        dataset_type: Either 'train' or 'val'.
        mask_types: List of mask types to create directories for.
    
    Returns:
        Dictionary of created paths with keys 'images' and 'masks_{type}'.
    """
    paths = {}
    
    # Images directory
    img_dir = Path(output_dir) / f'{dataset_type}_images' / dataset_type
    img_dir.mkdir(parents=True, exist_ok=True)
    paths['images'] = img_dir
    
    # Mask directories for each type
    for mask_type in mask_types:
        mask_dir = Path(output_dir) / f'{dataset_type}_masks_{mask_type}' / dataset_type
        mask_dir.mkdir(parents=True, exist_ok=True)
        paths[f'masks_{mask_type}'] = mask_dir
    
    return paths

def get_image_mask_pairs(data_dir, dataset_type, mask_types=['root', 'shoot', 'seed']):
    """Find all images and their corresponding masks.
    
    Args:
        data_dir: Root data directory (e.g., '../../data/dataset').
        dataset_type: Either 'train' or 'val'.
        mask_types: List of mask types to find.
    
    Returns:
        List of dictionaries with 'image' path and 'masks' dictionary.
        Each dict has structure: {'image': Path, 'masks': {'root': Path, ...}}
    """
    import glob
    
    image_dir = Path(data_dir) / f'{dataset_type}_images'
    mask_dir = Path(data_dir) / f'{dataset_type}_masks'
    
    # Get all image files
    image_files = sorted(glob.glob(str(image_dir / '*.png')))
    
    pairs = []
    for img_path in image_files:
        img_path = Path(img_path)
        base_name = img_path.stem  # filename without extension
        
        # Find corresponding masks
        masks = {}
        for mask_type in mask_types:
            mask_pattern = f'{base_name}_{mask_type}_mask.tif'
            mask_path = mask_dir / mask_pattern
            
            if mask_path.exists():
                masks[mask_type] = mask_path
        
        # Only include if at least one mask exists
        if masks:
            pairs.append({
                'image': img_path,
                'masks': masks
            })
    
    return pairs

def create_patches_from_image(image, mask_dict, patch_size, scaling_factor, step=None, 
                               roi_bbox=None, preprocess_fns=None):
    """Create patches from one image and its corresponding masks.
    
    Args:
        image: Numpy array of the image.
        mask_dict: Dictionary with mask_type: mask_array.
        patch_size: Size of patches.
        scaling_factor: Scaling factor for resizing.
        step: Step size for patch extraction. If None, defaults to patch_size (no overlap).
        roi_bbox: Optional ROI bounding box. If provided, crops image and masks before patching.
        preprocess_fns: List of preprocessing functions to apply before patching.
    
    Returns:
        Dictionary with 'image' patches, 'masks' dict of patches for each type, 
        'step', and 'roi_bbox' (if cropped).
    """
    
    if step is None:
        step = patch_size
    
    # Crop to ROI if provided
    if roi_bbox is not None:
        image = crop_to_roi(image, roi_bbox)
        mask_dict = {k: crop_to_roi(v, roi_bbox) for k, v in mask_dict.items()}
    
    # Apply preprocessing to image only (not masks)
    image = apply_preprocessing_pipeline(image, preprocess_fns)
    
    # Convert to grayscale if color image
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Process image (add channel dimension for grayscale)
    image = process_image(image, patch_size, scaling_factor, is_mask=False)
    image = image[..., np.newaxis]  # Add channel dimension for patchify
    img_patches = patchify(image, (patch_size, patch_size, 1), step=step)
    
    # Process masks
    mask_patches = {}
    for mask_type, mask in mask_dict.items():
        mask = process_image(mask, patch_size, scaling_factor, is_mask=True)
        mask = mask[..., np.newaxis]  # Add channel dimension for patchify
        patches = patchify(mask, (patch_size, patch_size, 1), step=step)
        mask_patches[mask_type] = patches
    
    result = {
        'image': img_patches,
        'masks': mask_patches,
        'step': step
    }
    
    if roi_bbox is not None:
        result['roi_bbox'] = roi_bbox
    
    return result

def reconstruct_from_patches(patches, image_shape, patch_size, step):
    """Reconstruct an image from patches.
    
    Uses unpatchify for non-overlapping patches (step == patch_size).
    Uses averaging reconstruction for overlapping patches (step < patch_size).
    
    Args:
        patches: Patch array from patchify with shape (n_rows, n_cols, 1, patch_size, patch_size, channels).
        image_shape: Target shape (height, width, channels) for reconstruction.
        patch_size: Size of each patch.
        step: Step size used during patch extraction.
    
    Returns:
        Reconstructed image.
    """
    # Use unpatchify for non-overlapping patches
    if step == patch_size:
        return unpatchify(patches, image_shape)
    
    # Use averaging for overlapping patches
    h, w, c = image_shape
    n_rows, n_cols = patches.shape[0], patches.shape[1]
    
    reconstructed = np.zeros(image_shape, dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            y_start = row_idx * step
            x_start = col_idx * step
            y_end = min(y_start + patch_size, h)
            x_end = min(x_start + patch_size, w)
            
            patch = patches[row_idx, col_idx, 0]
            patch_h = y_end - y_start
            patch_w = x_end - x_start
            
            reconstructed[y_start:y_end, x_start:x_end] += patch[:patch_h, :patch_w]
            counts[y_start:y_end, x_start:x_end] += 1
    
    counts = np.maximum(counts, 1)
    reconstructed = reconstructed / counts[:, :, np.newaxis]
    
    return reconstructed.astype(np.uint8)


def save_patches(pairs, output_dir, dataset_type, patch_size=128, 
                 scaling_factor=1.0, step=None, mask_types=['root', 'shoot', 'seed'],
                 filter_roi=True, preprocess_fns=None, notes=""):
    """Create and save all patches to disk, optionally cropping to ROI first.
    
    This function processes images serially. For faster processing with multiple
    images, use save_patches_parallel() instead.
    
    Args:
        pairs: List from get_image_mask_pairs().
        output_dir: Base output directory.
        dataset_type: Either 'train' or 'val'.
        patch_size: Patch size. Default is 128.
        scaling_factor: Scaling factor. Default is 1.0.
        step: Step size for patch extraction. If None, defaults to patch_size (no overlap).
        mask_types: Which masks to process. Default is ['root', 'shoot', 'seed'].
        filter_roi: If True, crop to ROI before patching. Default is True.
        preprocess_fns: Optional list of preprocessing functions to apply to images.
        notes: Optional notes to include in metadata. Default is empty string.
    
    Returns:
        Number of patches created.
    """
    # Simply delegate to save_patches_parallel with num_workers=1 for serial processing
    return save_patches_parallel(
        pairs=pairs,
        output_dir=output_dir,
        dataset_type=dataset_type,
        patch_size=patch_size,
        scaling_factor=scaling_factor,
        step=step,
        mask_types=mask_types,
        filter_roi=filter_roi,
        preprocess_fns=preprocess_fns,
        notes=notes,
        num_workers=1
    )

def _process_image_worker_parallel(args):
    """Worker function for parallel patch processing. Must be at module level for pickling.
    
    Args:
        args: Tuple of (pair, paths_dict, patch_size, scaling_factor, step, 
                       mask_types, filter_roi, preprocess_fns)
    
    Returns:
        Tuple of (local_metadata, patch_count, image_name)
    """
    pair, paths_dict, patch_size, scaling_factor, step, mask_types, filter_roi, preprocess_fns = args
    
    img_path = pair['image']
    base_name = img_path.stem
    
    # Load image as grayscale
    image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    # Detect ROI
    roi_bbox = detect_roi(image) if filter_roi else None
    
    # Load masks
    masks = {}
    for mask_type in mask_types:
        if mask_type in pair['masks']:
            masks[mask_type] = cv2.imread(str(pair['masks'][mask_type]), cv2.IMREAD_GRAYSCALE)
    
    # Create patches
    result = create_patches_from_image(image, masks, patch_size, scaling_factor, 
                                      step, roi_bbox, preprocess_fns)
    
    n_rows, n_cols = result['image'].shape[0], result['image'].shape[1]
    local_metadata = []
    
    # Write all patches for this image
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            patch_name = f"{base_name}_r{row_idx:02d}_c{col_idx:02d}.png"
            
            # Write image patch
            img_patch = result['image'][row_idx, col_idx, 0]
            cv2.imwrite(str(paths_dict['images'] / patch_name), img_patch)
            
            # Write mask patches
            for mask_type in mask_types:
                if mask_type in masks:
                    mask_patch = result['masks'][mask_type][row_idx, col_idx, 0]
                    cv2.imwrite(str(paths_dict[f'masks_{mask_type}'] / patch_name), mask_patch)
            
            # Calculate coordinates
            x_start = col_idx * step
            y_start = row_idx * step
            x_end = x_start + patch_size
            y_end = y_start + patch_size
            
            # Record metadata
            patch_metadata = {
                "patch_filename": patch_name,
                "source_image": img_path.name,
                "row_idx": row_idx,
                "col_idx": col_idx,
                "x_start": x_start,
                "y_start": y_start,
                "x_end": x_end,
                "y_end": y_end,
                "grid_size": [n_rows, n_cols]
            }
            
            if roi_bbox:
                patch_metadata["roi_bbox"] = [list(roi_bbox[0]), list(roi_bbox[1])]
            
            local_metadata.append(patch_metadata)
    
    return local_metadata, n_rows * n_cols, img_path.name

def save_patches_parallel(pairs, output_dir, dataset_type, patch_size=128, 
                         scaling_factor=1.0, step=None, mask_types=['root', 'shoot', 'seed'],
                         filter_roi=True, preprocess_fns=None, notes="", num_workers=None):
    """Create and save all patches to disk using parallel processing.
    
    This is an optimized version of save_patches() that processes multiple images
    in parallel using multiprocessing.
    
    Args:
        pairs: List from get_image_mask_pairs().
        output_dir: Base output directory.
        dataset_type: Either 'train' or 'val'.
        patch_size: Patch size. Default is 128.
        scaling_factor: Scaling factor. Default is 1.0.
        step: Step size for patch extraction. If None, defaults to patch_size (no overlap).
        mask_types: Which masks to process. Default is ['root', 'shoot', 'seed'].
        filter_roi: If True, crop to ROI before patching. Default is True.
        preprocess_fns: Optional list of preprocessing functions to apply to images.
        notes: Optional notes to include in metadata. Default is empty string.
        num_workers: Number of parallel workers. None = cpu_count() - 1. Default is None.
    
    Returns:
        Number of patches created.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    if step is None:
        step = patch_size
    
    # Auto-clean only directories for this dataset_type
    output_path = Path(output_dir)
    if output_path.exists():
        dirs_to_clean = [
            output_path / f'{dataset_type}_images',
            *[output_path / f'{dataset_type}_masks_{mt}' for mt in mask_types]
        ]
        
        for dir_path in dirs_to_clean:
            if dir_path.exists():
                print(f"Cleaning existing directory: {dir_path}")
                shutil.rmtree(dir_path)
    
    # Create directories
    paths = create_patch_directories(output_dir, dataset_type, mask_types)
    
    # Extract preprocessing function names for metadata
    preprocess_names = []
    if preprocess_fns:
        for fn in preprocess_fns:
            preprocess_names.append(fn.__name__)
    
    metadata = {
        "dataset_info": {
            "dataset_type": dataset_type,
            "dataset_source": str(pairs[0].get('image').parent.absolute().resolve()),
            "patch_size": patch_size,
            "step": step,
            "overlap_percent": (1 - step / patch_size) * 100,
            "scaling_factor": scaling_factor,
            "filter_roi": filter_roi,
            "preprocessing": preprocess_names if preprocess_names else None,
            "created_at": datetime.now().isoformat(),
            "epoch_utc": int(datetime.now(timezone.utc).timestamp()),            
            "num_source_images": len(pairs),
            "num_patches": 0,
            "notes": notes,
            "metadata_version": METADATA_VERSION
        },
        "patches": []
    }
    
    # Prepare arguments for all workers
    worker_args = [
        (pair, paths, patch_size, scaling_factor, step, mask_types, filter_roi, preprocess_fns)
        for pair in pairs
    ]
    
    # Process in parallel with progress bar
    patch_count = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process_image_worker_parallel, args): args[0]['image'].name 
                  for args in worker_args}
        
        with tqdm(total=len(pairs), desc=f"Processing images ({num_workers} workers)") as pbar:
            for future in as_completed(futures):
                try:
                    local_meta, count, img_name = future.result()
                    metadata['patches'].extend(local_meta)
                    patch_count += count
                    pbar.update(1)
                except Exception as e:
                    print(f"\nError processing {futures[future]}: {e}")
                    raise
    
    metadata['dataset_info']['num_patches'] = patch_count
    
    # Save metadata
    metadata_path = Path(output_dir) / f'{dataset_type}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nTotal: {patch_count} patches saved")
    print(f"Overlap: {metadata['dataset_info']['overlap_percent']:.1f}%")
    print(f"ROI cropping: {'enabled' if filter_roi else 'disabled'}")
    if preprocess_names:
        print(f"Preprocessing: {', '.join(preprocess_names)}")
    print(f"Workers used: {num_workers}")
    print(f"Metadata saved to {metadata_path}")
    
    return patch_count


def load_patch_metadata(patch_dir, dataset_type):
    """Load metadata for a saved patch dataset.
    
    Args:
        patch_dir: Directory containing saved patches.
        dataset_type: Either 'train' or 'val'.
    
    Returns:
        Dictionary containing dataset_info and patches list.
    
    Example:
        >>> metadata = load_patch_metadata('data/patched', 'train')
        >>> print(f"Patch size: {metadata['dataset_info']['patch_size']}")
        >>> print(f"Step size: {metadata['dataset_info']['step']}")
        >>> print(f"Total patches: {metadata['dataset_info']['num_patches']}")
    """
    metadata_path = Path(patch_dir) / f'{dataset_type}_metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return metadata

def get_patch_statistics(patch_dir, dataset_type):
    """Get statistics about a saved patch dataset.
    
    Args:
        patch_dir: Directory containing saved patches.
        dataset_type: Either 'train' or 'val'.
    
    Returns:
        Dictionary with dataset statistics.
    
    Example:
        >>> stats = get_patch_statistics('data/patched', 'train')
        >>> print(stats)
    """
    metadata = load_patch_metadata(patch_dir, dataset_type)
    info = metadata['dataset_info']
    
    stats = {
        'dataset_source': info['dataset_source'],
        'created_at': info['created_at'],
        'num_patches': info['num_patches'],
        'num_source_images': info['num_source_images'],
        'patch_size': info['patch_size'],
        'step': info['step'],
        'overlap_percent': info['overlap_percent'],
        'patches_per_image': info['num_patches'] / info['num_source_images']
    }
    
    return stats