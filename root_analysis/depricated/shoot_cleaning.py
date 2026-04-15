"""
Shoot Mask Cleaning Module

This module provides functions to clean and filter shoot segmentation masks
from ML model inference. The cleaning pipeline removes noise and identifies
individual plant locations using spatial and morphological filtering.

The pipeline consists of:
1. Morphological closing to connect fragmented components
2. Y-coordinate filtering to isolate the seed/shoot band
3. X-coordinate spacing filtering to identify regularly-spaced plants

Typical usage:
    from shoot_cleaning import clean_shoot_mask, analyze_shoot_zone_globally
    
    # First, analyze all masks to establish global zone parameters
    zone_stats = analyze_shoot_zone_globally(mask_file_list)
    zone_width = zone_stats['zone_width']
    
    # Then clean individual masks
    cleaned_mask, labels, stats = clean_shoot_mask(mask_path, zone_width)
"""

import cv2
import numpy as np
import skimage.measure
from pathlib import Path


def load_mask(mask_path):
    """Load a binary mask from a PNG file.
    
    Args:
        mask_path: Path to the mask file (str or Path object).
        
    Returns:
        Binary mask as numpy array with values 0 (background) and 255 (foreground).
        
    Raises:
        FileNotFoundError: If the mask file cannot be loaded.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not load mask from {mask_path}")
    return mask


def analyze_y_distribution(binary_mask):
    """Analyze the Y-coordinate distribution of foreground pixels in a mask.
    
    This function extracts statistics about where foreground pixels are located
    along the vertical (Y) axis. These statistics are used to identify the
    seed/shoot band location in the image.
    
    Args:
        binary_mask: Binary mask array with values 0 (background) and 255 (foreground).
        
    Returns:
        Dictionary containing:
            - min_y: Minimum Y-coordinate of foreground pixels
            - max_y: Maximum Y-coordinate of foreground pixels
            - mean_y: Mean Y-coordinate
            - median_y: Median Y-coordinate (typically where seeds/shoots are centered)
            - range: Vertical range (max_y - min_y)
            - y_coords: Array of all foreground Y-coordinates
            
        Returns None values if no foreground pixels are found.
    """
    y_coords, x_coords = np.where(binary_mask > 0)
    
    if len(y_coords) == 0:
        return {
            'min_y': None,
            'max_y': None,
            'mean_y': None,
            'median_y': None,
            'range': None,
            'y_coords': y_coords
        }
    
    stats = {
        'min_y': int(np.min(y_coords)),
        'max_y': int(np.max(y_coords)),
        'mean_y': float(np.mean(y_coords)),
        'median_y': float(np.median(y_coords)),
        'range': int(np.max(y_coords) - np.min(y_coords)),
        'y_coords': y_coords
    }
    
    return stats


def analyze_shoot_zone_globally(mask_paths, margin_std=2.0):
    """Analyze Y-coordinate statistics across multiple shoot masks to define a global shoot zone.
    
    This function processes multiple masks to establish consistent parameters for
    the seed/shoot band location. The zone width is calculated as the median ± margin_std
    standard deviations across all masks, capturing approximately 95% of typical
    shoot locations when margin_std=2.0.
    
    Args:
        mask_paths: List of paths to mask files (list of str or Path objects).
        margin_std: Number of standard deviations for zone width. Default is 2.0,
                   which captures ~95% of data in a normal distribution.
                   - Use 1.0 for tighter filtering (~68% coverage)
                   - Use 2.0 for balanced filtering (~95% coverage)
                   - Use 3.0 for more permissive filtering (~99.7% coverage)
    
    Returns:
        Dictionary containing:
            - median_of_medians: Central Y-position across all masks
            - mean_of_medians: Average Y-position of medians
            - std_of_medians: Standard deviation of median positions
            - min_median: Minimum median Y across all masks
            - max_median: Maximum median Y across all masks
            - zone_width: Calculated width of shoot zone (pixels)
            - global_zone: Tuple of (min_y, max_y) for the global zone
            - all_medians: List of median Y values from each mask
    """
    median_y_values = []
    
    for mask_path in mask_paths:
        mask = load_mask(mask_path)
        y_stats = analyze_y_distribution(mask)
        if y_stats['median_y'] is not None:
            median_y_values.append(y_stats['median_y'])
    
    if len(median_y_values) == 0:
        raise ValueError("No valid masks found for analysis")
    
    center = np.median(median_y_values)
    std = np.std(median_y_values)
    
    min_y = int(center - margin_std * std)
    max_y = int(center + margin_std * std)
    min_y = max(0, min_y)
    
    zone_stats = {
        'median_of_medians': float(center),
        'mean_of_medians': float(np.mean(median_y_values)),
        'std_of_medians': float(std),
        'min_median': float(np.min(median_y_values)),
        'max_median': float(np.max(median_y_values)),
        'zone_width': max_y - min_y,
        'global_zone': (min_y, max_y),
        'all_medians': median_y_values
    }
    
    return zone_stats


def get_adaptive_shoot_zone(y_stats, zone_width, margin=50):
    """Calculate an adaptive shoot zone centered on an image's own median Y-position.
    
    This function creates an image-specific shoot zone by centering a fixed-width
    band on the median Y-position of that particular image. This handles natural
    variation in seed tray positioning across different images.
    
    Args:
        y_stats: Y-distribution statistics from analyze_y_distribution().
        zone_width: Width of the zone in pixels (from analyze_shoot_zone_globally).
        margin: Additional pixels to add to each side of the zone for safety.
                Default is 50 pixels. Increase if legitimate shoots are being excluded.
        
    Returns:
        Tuple of (min_y, max_y) defining the adaptive shoot zone for this image.
    """
    median = y_stats['median_y']
    half_width = (zone_width / 2) + margin
    
    min_y = int(median - half_width)
    max_y = int(median + half_width)
    
    min_y = max(0, min_y)
    
    return (min_y, max_y)


def filter_components_by_zone(binary_mask, shoot_zone):
    """Filter connected components by keeping only those with centroids in the shoot zone.
    
    This function identifies separate connected components (blobs) in the mask and
    removes any components whose centroid falls outside the specified Y-coordinate range.
    This effectively removes root structures and noise that extend far below the seed band.
    
    Args:
        binary_mask: Binary mask array with values 0 and 255.
        shoot_zone: Tuple of (min_y, max_y) defining the acceptable Y-coordinate range.
        
    Returns:
        Tuple of (filtered_mask, filtered_labels):
            - filtered_mask: Binary mask containing only components in the zone
            - filtered_labels: Labeled image where each component has a unique integer ID
    """
    retval, labels = cv2.connectedComponents(binary_mask)
    regions = skimage.measure.regionprops(labels)
    
    keep_labels = []
    min_y, max_y = shoot_zone
    
    for region in regions:
        centroid_y = region.centroid[0]
        
        if min_y <= centroid_y <= max_y:
            keep_labels.append(region.label)
    
    filtered_labels = np.zeros_like(labels)
    for label in keep_labels:
        filtered_labels[labels == label] = label
    
    filtered_mask = (filtered_labels > 0).astype(np.uint8) * 255
    
    return filtered_mask, filtered_labels


def filter_by_spacing_with_size(labels, shoot_zone, expected_spacing, tolerance=0.5):
    """Filter components using regular X-spacing, keeping largest in each cluster.
    
    Plants in multi-well plates are typically arranged at regular X-intervals.
    This function exploits this spatial regularity to distinguish legitimate plants
    from noise. When multiple components are clustered close together (closer than
    the expected spacing), only the largest one is kept, as it's most likely the
    actual shoot while smaller ones are noise or seeds.
    
    The algorithm:
    1. Sort all components by X-position (left to right)
    2. Group nearby components into clusters based on proximity
    3. From each cluster, keep only the largest component
    4. This typically reduces ~20-40 components down to ~5 actual plants
    
    Args:
        labels: Labeled image from cv2.connectedComponents where each component
               has a unique integer ID.
        shoot_zone: Tuple of (min_y, max_y) to filter which components to consider.
                   Only components with centroids in this zone are analyzed.
        expected_spacing: Expected distance between adjacent plants in pixels.
                         For typical multi-well plates at microscopy resolution,
                         this is often 300-500 pixels.
        tolerance: Acceptable deviation from expected_spacing as a fraction.
                  Default 0.5 means ±50% of expected_spacing.
                  - 0.3 for stricter spacing requirements
                  - 0.5 for balanced filtering (recommended)
                  - 0.7 for more permissive spacing
        
    Returns:
        Tuple of (filtered_labels, keep_labels):
            - filtered_labels: Labeled image containing only kept components
            - keep_labels: List of component IDs that were kept
    """
    regions = skimage.measure.regionprops(labels)
    
    # Get components in shoot zone
    min_y, max_y = shoot_zone
    components = []
    
    for region in regions:
        centroid_y, centroid_x = region.centroid
        if min_y <= centroid_y <= max_y:
            components.append({
                'x': centroid_x,
                'y': centroid_y,
                'label': region.label,
                'area': region.area
            })
    
    if len(components) == 0:
        return np.zeros_like(labels), []
    
    # Sort by X position (left to right)
    components.sort(key=lambda c: c['x'])
    
    min_spacing = expected_spacing * (1 - tolerance)
    
    # Group components into clusters based on proximity
    clusters = []
    current_cluster = [components[0]]
    
    for comp in components[1:]:
        spacing = comp['x'] - current_cluster[-1]['x']
        
        # If close to last component, add to current cluster
        if spacing < min_spacing:
            current_cluster.append(comp)
        else:
            # Start new cluster
            clusters.append(current_cluster)
            current_cluster = [comp]
    
    # Don't forget the last cluster
    clusters.append(current_cluster)
    
    # From each cluster, keep the largest component (likely the shoot, not noise/seed)
    keep_labels = []
    for cluster in clusters:
        largest = max(cluster, key=lambda c: c['area'])
        keep_labels.append(largest['label'])
    
    # Create filtered labels
    filtered_labels = np.zeros_like(labels)
    for label in keep_labels:
        filtered_labels[labels == label] = label
    
    return filtered_labels, keep_labels


def clean_shoot_mask(mask_path, zone_width, margin=50, 
                     expected_spacing=400, tolerance=0.5,
                     closing_kernel_size=5, closing_iterations=2):
    """Complete shoot mask cleaning pipeline.
    
    This is the main entry point for cleaning a single shoot mask. It applies
    a multi-stage pipeline to remove noise and identify individual plant locations:
    
    1. Morphological closing: Fills small gaps within shoots to connect fragmented parts
    2. Y-zone filtering: Removes components outside the seed/shoot band
    3. X-spacing filtering: Uses regular plant spacing to distinguish plants from noise
    
    The pipeline typically reduces noisy masks with 20-70 components down to ~5 clean
    plant locations, corresponding to the actual plants in the multi-well plate.
    
    Args:
        mask_path: Path to shoot mask file (str or Path).
                  Mask should be a binary PNG with 0 (background) and 255 (foreground).
        zone_width: Width of shoot zone in pixels, obtained from analyze_shoot_zone_globally().
                   This represents the typical vertical extent of the seed/shoot band.
        margin: Additional pixels to add above/below the adaptive zone for safety.
               Default 50 pixels. Increase if legitimate shoots are being excluded.
        expected_spacing: Expected horizontal distance between adjacent plants in pixels.
                         Default 400 pixels works for plates 
        tolerance: Acceptable spacing deviation as fraction of expected_spacing.
                  Default 0.5 allows +/-50% variation in plant spacing.
        closing_kernel_size: Size of square structuring element for morphological closing.
                            Default 5 pixels. Common values:
                            - 3: Minimal closing, preserves fine details
                            - 5: Balanced closing (recommended)
                            - 7: Aggressive closing, may merge nearby components
        closing_iterations: Number of times to apply morphological closing.
                           Default 2. More iterations = more aggressive gap filling.
        
    Returns:
        Tuple of (cleaned_mask, cleaned_labels, stats):
            - cleaned_mask: Binary mask (0/255) containing only cleaned shoot components
            - cleaned_labels: Labeled image where each plant has a unique integer ID
            - stats: Dictionary containing:
                * adaptive_zone: (min_y, max_y) zone used for this image
                * num_original: Component count before cleaning
                * num_after_closing: Component count after morphological closing
                * num_after_y_filter: Component count after Y-zone filtering
                * num_final: Final component count (typically ~5)
                * closing_params: String describing closing parameters used
    
    Example:
        # First establish global zone parameters
        mask_files = list(Path('masks/').glob('*.png'))
        zone_stats = analyze_shoot_zone_globally(mask_files)
        zone_width = zone_stats['zone_width']
        
        # Then clean individual masks
        for mask_file in mask_files:
            cleaned_mask, labels, stats = clean_shoot_mask(mask_file, zone_width)
            print(f"{mask_file.name}: {stats['num_original']} -> {stats['num_final']} components")
    """
    # Load mask
    mask = load_mask(mask_path)
    
    # Apply morphological closing to fill gaps and connect fragmented components
    kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                   iterations=closing_iterations)
    
    # Get adaptive shoot zone based on this image's Y-distribution
    y_stats = analyze_y_distribution(closed_mask)
    adaptive_zone = get_adaptive_shoot_zone(y_stats, zone_width, margin=margin)
    
    # Filter by Y-zone to remove components outside shoot band
    y_filtered_mask, y_filtered_labels = filter_components_by_zone(
        closed_mask, adaptive_zone
    )
    
    # Filter by X-spacing, keeping largest component in each spatial cluster
    final_labels, keep_labels = filter_by_spacing_with_size(
        y_filtered_labels, adaptive_zone, expected_spacing, tolerance
    )
    final_mask = (final_labels > 0).astype(np.uint8) * 255
    
    # Compile statistics about the cleaning process
    stats = {
        'adaptive_zone': adaptive_zone,
        'num_original': len(np.unique(cv2.connectedComponents(mask)[1])) - 1,
        'num_after_closing': len(np.unique(cv2.connectedComponents(closed_mask)[1])) - 1,
        'num_after_y_filter': len(np.unique(y_filtered_labels)) - 1,
        'num_final': len(keep_labels),
        'closing_params': f'kernel={closing_kernel_size}x{closing_kernel_size}, iter={closing_iterations}'
    }
    
    return final_mask, final_labels, stats