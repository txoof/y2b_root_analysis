# library/dataset_visualization.py
"""Visualization and testing functions for patch datasets."""

import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from library.roi import detect_roi, crop_to_roi
from library.patch_dataset import create_patches_from_image


def verify_patches(pairs, patch_size=128, scaling_factor=1.0, step=None,
                   mask_type='root', filter_roi=True, preprocess_fns=None):
    """Display a random patch with original image (gridded), mask, and overlay for verification.
    
    Args:
        pairs: List of image-mask pairs from get_image_mask_pairs().
        patch_size: Size of patches. Default is 128.
        scaling_factor: Scaling factor. Default is 1.0.
        step: Step size for patch extraction. If None, defaults to patch_size (no overlap).
        mask_type: Which mask to display ('root', 'shoot', or 'seed'). Default is 'root'.
        filter_roi: If True, crop to ROI before patching. Default is True.
        preprocess_fns: Optional list of preprocessing functions to apply to image.
    
    Returns:
        None
    """
    if step is None:
        step = patch_size
    
    # Color mapping for mask types
    color_map = {
        'shoot': {'rgb': [0, 255, 0], 'name': 'green'},    # Green
        'seed': {'rgb': [0, 0, 255], 'name': 'blue'},      # Blue
        'root': {'rgb': [255, 0, 0], 'name': 'red'},       # Red
    }
    mask_colors = color_map.get(mask_type, {'rgb': [255, 255, 0], 'name': 'yellow'})

    # Pick random image
    pair = random.choice(pairs)
    img_path = pair['image']
    
    # Load image and mask
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(pair['masks'][mask_type]), cv2.IMREAD_GRAYSCALE)
    
    # Detect ROI
    roi_bbox = detect_roi(image) if filter_roi else None
    
    # Create patches
    masks_dict = {mask_type: mask}
    result = create_patches_from_image(image, masks_dict, patch_size, scaling_factor, 
                                      step, roi_bbox, preprocess_fns)
    
    # Pick random patch that has mask content
    n_rows, n_cols = result['image'].shape[0], result['image'].shape[1]
    
    valid_patches = []
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            patch_mask = result['masks'][mask_type][row_idx, col_idx, 0, :, :, 0]
            if patch_mask.sum() > 0:
                valid_patches.append((row_idx, col_idx))
    
    if not valid_patches:
        print(f"No valid patches found for {img_path.name}")
        return
    
    # Pick random valid patch
    row_idx, col_idx = random.choice(valid_patches)
    
    # Extract patches
    img_patch = result['image'][row_idx, col_idx, 0]
    mask_patch = result['masks'][mask_type][row_idx, col_idx, 0, :, :, 0]
    
    # Calculate patch center (in cropped coordinates if ROI used)
    center_y = row_idx * step + patch_size // 2
    center_x = col_idx * step + patch_size // 2
    
    # Create overlay
    overlay = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB).copy()
    overlay[mask_patch > 0] = mask_colors['rgb']
    
    # Prepare display image (cropped if ROI used)
    display_image = crop_to_roi(image, roi_bbox) if roi_bbox else image
    # Apply preprocessing for display
    if preprocess_fns:
        from library.patch_dataset import apply_preprocessing_pipeline
        display_image = apply_preprocessing_pipeline(display_image, preprocess_fns)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original/cropped image with grid
    axes[0, 0].imshow(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
    for i in range(0, display_image.shape[0], step):
        axes[0, 0].axhline(y=i, color=mask_colors['name'], linewidth=0.5, alpha=0.3)
    for j in range(0, display_image.shape[1], step):
        axes[0, 0].axvline(x=j, color=mask_colors['name'], linewidth=0.5, alpha=0.3)
    
    axes[0, 0].plot(center_x, center_y, 'r*', markersize=15)
    title = 'Cropped to ROI' if roi_bbox else 'Original Image'
    if preprocess_fns:
        title += ' (preprocessed)'
    axes[0, 0].set_title(title)
    axes[0, 0].axis('off')
    
    # Patch
    axes[0, 1].imshow(cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'Patch [{row_idx},{col_idx}]')
    axes[0, 1].axis('off')
    
    # Mask
    axes[1, 0].imshow(mask_patch, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title('Mask')
    axes[1, 0].axis('off')
    
    # Overlay
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay')
    axes[1, 1].axis('off')
    
    overlap_pct = (1 - step/patch_size) * 100
    title = f'{img_path.name} - {mask_type} (overlap {overlap_pct:.0f}%)'
    title += f' - {len(valid_patches)} valid patches'
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def test_unpatchify(pairs, img_idx=None, mask_type='root', patch_size=128, 
                    scaling_factor=1.0, step=None, filter_roi=True, preprocess_fns=None):
    """Test that patches can be reconstructed back into the original image.
    
    Args:
        pairs: List of image-mask pairs.
        img_idx: Index of image to test. If None, picks random. Default is None.
        mask_type: Which mask to test. Default is 'root'.
        patch_size: Size of patches. Default is 128.
        scaling_factor: Scaling factor. Default is 1.0.
        step: Step size for patch extraction. If None, defaults to patch_size (no overlap).
        filter_roi: If True, crop to ROI before patching. Default is True.
        preprocess_fns: Optional list of preprocessing functions to apply to image.
    
    Returns:
        Boolean indicating if reconstruction matches original.
    """
    from library.patch_dataset import process_image, reconstruct_from_patches
    
    if step is None:
        step = patch_size
    
    # Pick image
    if img_idx is None:
        pair = random.choice(pairs)
    else:
        pair = pairs[img_idx]
    
    # Load image and mask
    img_path = pair['image']
    image = cv2.imread(str(img_path))
    mask = cv2.imread(str(pair['masks'][mask_type]), cv2.IMREAD_GRAYSCALE)
    
    # Detect ROI
    roi_bbox = detect_roi(image) if filter_roi else None
    
    print(f"Testing: {img_path.name}")
    print(f"Original image shape: {image.shape}")
    print(f"Original mask shape: {mask.shape}")
    if roi_bbox:
        print(f"ROI bbox: {roi_bbox}")
    if preprocess_fns:
        print(f"Preprocessing: {[fn.__name__ for fn in preprocess_fns]}")
    print(f"Patch size: {patch_size}, Step: {step}, Overlap: {(1 - step/patch_size)*100:.1f}%")
    
    # Create patches
    masks_dict = {mask_type: mask}
    result = create_patches_from_image(image, masks_dict, patch_size, scaling_factor, 
                                      step, roi_bbox, preprocess_fns)
    
    # Get patched shapes
    img_patches = result['image']
    mask_patches = result['masks'][mask_type]
    
    print(f"Patches shape: {img_patches.shape}")
    
    # Get padded dimensions
    n_rows, n_cols = img_patches.shape[0], img_patches.shape[1]
    padded_h = n_rows * step + (patch_size - step)
    padded_w = n_cols * step + (patch_size - step)
    
    print(f"Padded dimensions: {padded_h} x {padded_w}")
    
    # Get the image we should be reconstructing (cropped if ROI used)
    target_image = crop_to_roi(image, roi_bbox) if roi_bbox else image
    target_mask = crop_to_roi(mask, roi_bbox) if roi_bbox else mask
    
    # Apply preprocessing to target for comparison
    if preprocess_fns:
        from library.patch_dataset import apply_preprocessing_pipeline
        target_image = apply_preprocessing_pipeline(target_image, preprocess_fns)
    
    # Process target for comparison (same padding applied during patching)
    processed_img = process_image(target_image, patch_size, scaling_factor, is_mask=False)
    processed_mask = process_image(target_mask, patch_size, scaling_factor, is_mask=True)
    
    # Reconstruct
    reconstructed_img = reconstruct_from_patches(
        img_patches, (padded_h, padded_w, 3), patch_size, step
    )
    reconstructed_mask = reconstruct_from_patches(
        mask_patches, (padded_h, padded_w, 1), patch_size, step
    )
    
    print(f"Reconstructed image shape: {reconstructed_img.shape}")
    print(f"Reconstructed mask shape: {reconstructed_mask.shape}")
    
    # Check if reconstruction matches processed original
    img_match = np.array_equal(reconstructed_img, processed_img)
    mask_match = np.array_equal(reconstructed_mask[:, :, 0], processed_mask)
    
    print(f"Image reconstruction matches: {img_match}")
    print(f"Mask reconstruction matches: {mask_match}")
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original with ROI
    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if roi_bbox:
        (x1, y1), (x2, y2) = roi_bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                              linewidth=2, edgecolor='green', facecolor='none')
        axes[0, 0].add_patch(rect)
    axes[0, 0].set_title('Original Image' + (' + ROI' if roi_bbox else ''))
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray')
    axes[1, 0].set_title('Original Mask')
    axes[1, 0].axis('off')
    
    # Padded/Processed (cropped + preprocessed if ROI)
    axes[0, 1].imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
    title = 'Cropped + Preprocessed + Padded' if roi_bbox and preprocess_fns else \
            'Cropped + Padded' if roi_bbox else \
            'Preprocessed + Padded' if preprocess_fns else 'Padded Image'
    axes[0, 1].set_title(title)
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(processed_mask, cmap='gray')
    axes[1, 1].set_title('Padded Mask')
    axes[1, 1].axis('off')
    
    # Reconstructed
    axes[0, 2].imshow(cv2.cvtColor(reconstructed_img, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title(f'Reconstructed (overlap {(1-step/patch_size)*100:.0f}%)')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(reconstructed_mask[:, :, 0], cmap='gray')
    axes[1, 2].set_title('Reconstructed Mask')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return img_match and mask_match

def visualize_patch_reassembly(patch_dir, raw_dir, dataset_type, n_images=3, 
                               mask_types=['root', 'shoot', 'seed']):
    """Visualize original images alongside their reassembled patches and mask overlays.
    
    Loads random original images and their corresponding patches, reassembles
    the patches, and displays them side-by-side with yellow borders showing
    patch boundaries. Also shows mask overlays in different colors.
    
    Args:
        patch_dir: Directory containing saved patches (e.g., '../../data/test_output/patched_test').
        raw_dir: Directory containing original images (e.g., '../../data/dataset').
        dataset_type: Either 'train' or 'val'.
        n_images: Number of random images to visualize. Default is 3.
        mask_types: List of mask types to overlay. Default is ['root', 'shoot', 'seed'].
    
    Example:
        >>> visualize_patch_reassembly('../../data/test_output/patched_test', 
        ...                            '../../data/dataset', 'train', n_images=3)
    """
    from library.patch_dataset import load_patch_metadata, reconstruct_from_patches
    
    # Define colors for each mask type (RGB)
    mask_color_map = {
        'root': [255, 0, 0],      # Red
        'shoot': [0, 255, 0],     # Green
        'seed': [0, 0, 255]       # Blue
    }
    
    # Load metadata
    metadata = load_patch_metadata(patch_dir, dataset_type)
    patch_size = metadata['dataset_info']['patch_size']
    step = metadata['dataset_info']['step']
    filter_roi = metadata['dataset_info'].get('filter_roi', False)
    preprocessing = metadata['dataset_info'].get('preprocessing', None)
    
    # Get unique source images
    source_images = list(set([p['source_image'] for p in metadata['patches']]))
    
    # Select n random images
    selected_images = random.sample(source_images, min(n_images, len(source_images)))
    
    for img_name in selected_images:
        print(f"Processing {img_name}...")
        
        # Load original image
        img_path = Path(raw_dir) / f'{dataset_type}_images' / img_name
        original_image = cv2.imread(str(img_path))
        
        if original_image is None:
            print(f"Warning: Could not load {img_path}")
            continue
        
        # Get all patches for this image
        image_patches = [p for p in metadata['patches'] if p['source_image'] == img_name]
        
        if not image_patches:
            print(f"Warning: No patches found for {img_name}")
            continue
        
        # Get ROI bbox from first patch (all patches from same image have same ROI)
        roi_bbox = None
        if 'roi_bbox' in image_patches[0]:
            roi_bbox = tuple(tuple(coord) for coord in image_patches[0]['roi_bbox'])
        
        # Determine grid size from metadata
        grid_size = image_patches[0]['grid_size']
        n_rows, n_cols = grid_size
        
        # Calculate padded dimensions
        padded_h = n_rows * step + (patch_size - step)
        padded_w = n_cols * step + (patch_size - step)
        
        # Initialize arrays for reassembled image and masks
        if step < patch_size:
            # Use averaging reconstruction for overlapping patches
            reassembled = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
            counts = np.zeros((padded_h, padded_w), dtype=np.float32)
            
            reassembled_masks = {}
            mask_counts = {}
            for mask_type in mask_types:
                reassembled_masks[mask_type] = np.zeros((padded_h, padded_w), dtype=np.float32)
                mask_counts[mask_type] = np.zeros((padded_h, padded_w), dtype=np.float32)
        else:
            # Simple unpatchify for non-overlapping
            reassembled = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
            reassembled_masks = {}
            for mask_type in mask_types:
                reassembled_masks[mask_type] = np.zeros((padded_h, padded_w), dtype=np.uint8)
        
        # Load and place each patch
        patch_coords = []
        for patch_info in image_patches:
            row_idx = patch_info['row_idx']
            col_idx = patch_info['col_idx']
            patch_filename = patch_info['patch_filename']
            
            # Load image patch
            patch_path = Path(patch_dir) / f'{dataset_type}_images' / dataset_type / patch_filename
            patch = cv2.imread(str(patch_path))
            
            if patch is None:
                continue
            
            # Calculate position
            y_start = row_idx * step
            x_start = col_idx * step
            y_end = min(y_start + patch_size, padded_h)
            x_end = min(x_start + patch_size, padded_w)
            
            patch_h = y_end - y_start
            patch_w = x_end - x_start
            
            # Place image patch
            if step < patch_size:
                # Averaging for overlapping patches
                reassembled[y_start:y_end, x_start:x_end] += patch[:patch_h, :patch_w]
                counts[y_start:y_end, x_start:x_end] += 1
            else:
                # Direct placement for non-overlapping
                reassembled[y_start:y_end, x_start:x_end] = patch[:patch_h, :patch_w]
            
            # Load and place mask patches
            for mask_type in mask_types:
                mask_patch_path = Path(patch_dir) / f'{dataset_type}_masks_{mask_type}' / dataset_type / patch_filename
                if mask_patch_path.exists():
                    mask_patch = cv2.imread(str(mask_patch_path), cv2.IMREAD_GRAYSCALE)
                    if mask_patch is not None:
                        if step < patch_size:
                            reassembled_masks[mask_type][y_start:y_end, x_start:x_end] += mask_patch[:patch_h, :patch_w]
                            mask_counts[mask_type][y_start:y_end, x_start:x_end] += 1
                        else:
                            reassembled_masks[mask_type][y_start:y_end, x_start:x_end] = mask_patch[:patch_h, :patch_w]
            
            # Store coordinates for drawing borders
            patch_coords.append((x_start, y_start, x_end, y_end))
        
        # Average overlapping regions if needed
        if step < patch_size:
            counts = np.maximum(counts, 1)
            reassembled = reassembled / counts[:, :, np.newaxis]
            reassembled = reassembled.astype(np.uint8)
            
            for mask_type in mask_types:
                mask_counts[mask_type] = np.maximum(mask_counts[mask_type], 1)
                reassembled_masks[mask_type] = reassembled_masks[mask_type] / mask_counts[mask_type]
                reassembled_masks[mask_type] = reassembled_masks[mask_type].astype(np.uint8)
        
        # Create mask overlay on reassembled image
        overlay = cv2.cvtColor(reassembled, cv2.COLOR_BGR2RGB).copy()
        
        # Apply each mask with its color
        for mask_type in mask_types:
            if mask_type in reassembled_masks:
                mask = reassembled_masks[mask_type]
                color = mask_color_map.get(mask_type, [255, 255, 255])
                overlay[mask > 0] = color
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image (with ROI if available)
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        if roi_bbox:
            (x1, y1), (x2, y2) = roi_bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                linewidth=2, edgecolor='green', facecolor='none')
            axes[0].add_patch(rect)
        title = f'Original Image\n{img_name}'
        if roi_bbox:
            title += '\n(green = ROI used for patching)'
        axes[0].set_title(title)
        axes[0].axis('off')
        
        # Reassembled image with patch borders
        axes[1].imshow(cv2.cvtColor(reassembled, cv2.COLOR_BGR2RGB))
        
        # Draw yellow borders for each patch
        for x_start, y_start, x_end, y_end in patch_coords:
            rect = plt.Rectangle(
                (x_start, y_start), 
                x_end - x_start, 
                y_end - y_start,
                linewidth=1, 
                edgecolor='yellow', 
                facecolor='none'
            )
            axes[1].add_patch(rect)
        
        title = f'Reassembled from {len(patch_coords)} Patches\n'
        title += f'Patch size: {patch_size}, Step: {step} '
        title += f'(overlap: {(1-step/patch_size)*100:.0f}%)'
        if filter_roi:
            title += '\n(cropped to ROI before patching)'
        if preprocessing:
            title += f'\nPreprocessing: {", ".join(preprocessing)}'
        axes[1].set_title(title)
        axes[1].axis('off')
        
        # Mask overlay
        axes[2].imshow(overlay)
        
        # Create legend for mask colors
        legend_text = []
        for mask_type in mask_types:
            if mask_type in reassembled_masks:
                color_name = mask_type.capitalize()
                legend_text.append(f'{color_name}: {mask_color_map[mask_type]}')
        
        axes[2].set_title(f'Mask Overlay\n' + ', '.join(legend_text))
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"  Reassembled {len(patch_coords)} patches into {padded_h}x{padded_w} image")
        if roi_bbox:
            print(f"  Original: {original_image.shape[1]}x{original_image.shape[0]}, Cropped to ROI before patching")
        else:
            print(f"  Original: {original_image.shape[1]}x{original_image.shape[0]}")
        if preprocessing:
            print(f"  Preprocessing applied: {', '.join(preprocessing)}")
        print()
