"""Shoot mask cleaning pipeline for plant root analysis.

This module provides functions for cleaning and filtering shoot segmentation masks
from U-Net model predictions. The pipeline identifies exactly 5 shoot locations
using adaptive Y-zone detection and X-axis peak finding, handling edge cases like
failed germination and fallen shoots.

Typical workflow:
    
    Step 1: Calculate global statistics across all masks (one-time setup)
    
        >>> from pathlib import Path
        >>> import cv2
        >>> from shoot_mask_cleaning import calculate_global_y_stats
        >>> 
        >>> mask_dir = Path('data/shoot_masks')
        >>> mask_paths = [str(f) for f in sorted(mask_dir.glob('*.png'))]
        >>> 
        >>> global_stats = calculate_global_y_stats(
        ...     mask_paths, 
        ...     kernel_size=5,
        ...     iterations=3,
        ...     y_min=200, 
        ...     y_max=750
        ... )
        >>> print(f"Global mean Y: {global_stats['global_mean']:.1f}")
        >>> print(f"Global std Y: {global_stats['global_std']:.1f}")
    
    Step 2: Process individual masks
    
        >>> from shoot_mask_cleaning import clean_shoot_mask_pipeline
        >>> 
        >>> for mask_path in mask_paths:
        ...     result = clean_shoot_mask_pipeline(mask_path, global_stats)
        ...     
        ...     # Save cleaned mask
        ...     output_path = f"cleaned/{result['filename']}"
        ...     cv2.imwrite(output_path, result['cleaned_mask'])
        ...     
        ...     # Check results
        ...     print(f"{result['filename']}: {result['method']}, "
        ...           f"{result['num_components']} components")
    
    Step 3: Debug problematic images
    
        >>> from shoot_mask_visualization import debug_peak_detection
        >>> 
        >>> # Visualize complete pipeline for troubleshooting
        >>> debug_peak_detection('data/shoot_masks/problem_image.png', global_stats)

Recommended parameters (validated on 19 test images):
    
    Initial closing:
        - kernel_size: 7
        - iterations: 5
    
    Y-zone boundaries:
        - y_min: 200 (shoots never above this)
        - y_max: 750 (shoots never below this)
    
    Peak detection:
        - n_peaks: 5 (expected number of plants)
        - min_distance: 300 (minimum spacing between shoots)
        - x_min: 1000 (shoots never left of this)
        - initial_std: 2.0 (starting Y-zone width)
        - quality_check_threshold: 2.5 (when to validate peak quality)
        - min_peak_width: 20 (minimum width for valid peaks)
        - min_peak_height: 10 (minimum height for valid peaks)
        - min_area: 500 (for size-based fallback)
    
    Filtering:
        - band_width: 100 (X-distance around peaks)

Dependencies:
    - numpy
    - opencv-python (cv2)
    - scipy (for peak detection)

Author: Aaron Ciuffo
Date: December 2024
"""

import numpy as np
import cv2
from pathlib import Path
from scipy.signal import find_peaks, peak_widths


def join_shoot_fragments(mask, kernel_size=7, iterations=5):
    """Join small shoot fragments using morphological closing.
    
    Applies morphological closing to connect nearby fragments and fill small gaps
    in the segmentation mask. This preprocessing step improves peak detection by
    creating more continuous shoot structures.
    
    Args:
        mask: Binary mask array with values 0 and 255.
        kernel_size: Size of square structuring element (default: 7).
        iterations: Number of closing iterations (default: 5).
        
    Returns:
        Binary mask array with joined fragments (values 0 and 255).
        
    Example:
        >>> mask = cv2.imread('shoot_mask.png', cv2.IMREAD_GRAYSCALE)
        >>> joined = join_shoot_fragments(mask, kernel_size=5, iterations=3)
        >>> cv2.imwrite('joined_mask.png', joined)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return joined


def calculate_y_density(mask):
    """Calculate pixel density per Y coordinate.
    
    Sums the number of foreground pixels along each row to create a 1D density
    profile. Used for finding the vertical position of shoots.
    
    Args:
        mask: Binary mask with values 0 and 255.
        
    Returns:
        1D numpy array of pixel counts per row (length = mask height).
        
    Example:
        >>> mask = cv2.imread('shoot_mask.png', cv2.IMREAD_GRAYSCALE)
        >>> density = calculate_y_density(mask)
        >>> print(f"Peak density at Y={np.argmax(density)}")
    """
    return np.sum(mask > 0, axis=1)


def calculate_weighted_y_stats(density, y_min=200, y_max=750):
    """Calculate weighted mean and standard deviation of Y positions within ROI.
    
    Computes the center of mass and spread of shoot pixels along the Y-axis,
    restricted to the valid Y-range. Pixels outside the ROI are ignored to
    prevent noise from biasing statistics.
    
    Args:
        density: 1D array of pixel counts per row from calculate_y_density.
        y_min: Minimum Y coordinate to consider (default: 200).
        y_max: Maximum Y coordinate to consider (default: 750).
        
    Returns:
        Dictionary with keys:
            - 'mean': Weighted mean Y position (pixels).
            - 'std': Weighted standard deviation (pixels).
        
    Example:
        >>> density = calculate_y_density(mask)
        >>> stats = calculate_weighted_y_stats(density, y_min=200, y_max=750)
        >>> print(f"Center of mass: Y={stats['mean']:.1f} ± {stats['std']:.1f}")
    """
    # Clip density to ROI
    roi_density = density.copy()
    roi_density[:y_min] = 0
    roi_density[y_max:] = 0
    
    y_positions = np.arange(len(roi_density))
    total_pixels = np.sum(roi_density)
    
    if total_pixels == 0:
        return {'mean': 0, 'std': 0}
    
    # Weighted mean (center of mass)
    weighted_mean = np.sum(y_positions * roi_density) / total_pixels
    
    # Weighted standard deviation
    weighted_var = np.sum(roi_density * (y_positions - weighted_mean)**2) / total_pixels
    weighted_std = np.sqrt(weighted_var)
    
    return {
        'mean': weighted_mean,
        'std': weighted_std
    }


def calculate_global_y_stats(mask_paths, kernel_size=7, iterations=5, y_min=200, y_max=750):
    """Calculate global weighted mean and standard deviation across all masks.
    
    Processes all masks to establish global shoot position statistics. These
    statistics serve as priors for individual image processing, enabling adaptive
    Y-zone detection that handles variation in shoot positions.
    
    Args:
        mask_paths: List of paths (strings or Path objects) to mask files.
        kernel_size: Kernel size for initial morphological closing (default: 7).
        iterations: Number of closing iterations (default: 5).
        y_min: Minimum Y coordinate to consider (default: 200).
        y_max: Maximum Y coordinate to consider (default: 750).
        
    Returns:
        Dictionary with keys:
            - 'global_mean': Mean Y position across all images (pixels).
            - 'global_std': Mean standard deviation across all images (pixels).
            - 'all_stats': List of per-image statistics dictionaries.
            - 'all_means': List of individual image means.
            - 'all_stds': List of individual image standard deviations.
            - 'y_min': Y minimum boundary used.
            - 'y_max': Y maximum boundary used.
        
    Example:
        >>> mask_files = [str(f) for f in Path('data/shoot').glob('*.png')]
        >>> global_stats = calculate_global_y_stats(mask_files, y_min=200, y_max=750)
        >>> print(f"Expected shoot zone: {global_stats['global_mean']:.0f} ± "
        ...       f"{2 * global_stats['global_std']:.0f} pixels")
    
    Note:
        This function should be called once per dataset to establish the global
        statistics used for processing all individual images.
    """
    all_stats = []
    
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        joined = join_shoot_fragments(mask, kernel_size, iterations)
        density = calculate_y_density(joined)
        stats = calculate_weighted_y_stats(density, y_min, y_max)
        all_stats.append(stats)
    
    # Calculate mean of means and mean of stds
    means = [s['mean'] for s in all_stats]
    stds = [s['std'] for s in all_stats]
    
    global_mean = np.mean(means)
    global_std = np.mean(stds)
    
    return {
        'global_mean': global_mean,
        'global_std': global_std,
        'all_stats': all_stats,
        'all_means': means,
        'all_stds': stds,
        'y_min': y_min,
        'y_max': y_max
    }


def calculate_x_projection(mask, global_stats, std_multiplier=2.0):
    """Calculate X-axis projection within the global Y-zone.
    
    Sums pixels along columns (Y-axis) within the vertical zone defined by global
    statistics. Creates a 1D signal showing horizontal distribution of shoot mass.
    
    Args:
        mask: Binary mask with values 0 and 255.
        global_stats: Global statistics from calculate_global_y_stats.
        std_multiplier: Standard deviation multiplier for Y-zone width (default: 2.0).
        
    Returns:
        1D numpy array of pixel counts per column (length = mask width).
        
    Example:
        >>> x_proj = calculate_x_projection(mask, global_stats, std_multiplier=2.0)
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(x_proj)
        >>> plt.xlabel('X coordinate')
        >>> plt.ylabel('Pixel count')
        >>> plt.show()
    """
    # Calculate Y bounds
    global_upper = int(global_stats['global_mean'] - std_multiplier * global_stats['global_std'])
    global_lower = int(global_stats['global_mean'] + std_multiplier * global_stats['global_std'])
    
    # Extract Y-zone region and sum along Y axis
    zone_region = mask[global_upper:global_lower, :]
    x_projection = np.sum(zone_region > 0, axis=0)
    
    return x_projection


def find_shoot_peaks(x_projection, n_peaks=5, min_distance=300, x_min=1000):
    """Find peak locations in X-projection using scipy peak detection.
    
    Identifies the N strongest peaks in the horizontal projection that satisfy
    spacing and position constraints. Returns peaks sorted left to right.
    
    Args:
        x_projection: 1D array of pixel counts from calculate_x_projection.
        n_peaks: Number of peaks to find (default: 5).
        min_distance: Minimum distance between peaks in pixels (default: 300).
        x_min: Minimum X coordinate to consider (default: 1000).
        
    Returns:
        Numpy array of peak X-coordinates, sorted left to right.
        
    Example:
        >>> x_proj = calculate_x_projection(mask, global_stats)
        >>> peaks = find_shoot_peaks(x_proj, n_peaks=5, min_distance=300)
        >>> print(f"Found peaks at X positions: {peaks}")
    """
    # Mask out x < x_min
    masked_projection = x_projection.copy()
    masked_projection[:x_min] = 0
    
    # Find peaks with minimum distance constraint
    peaks, properties = find_peaks(masked_projection, distance=min_distance)
    
    # Get peak heights and select top n_peaks
    peak_heights = masked_projection[peaks]
    top_indices = np.argsort(peak_heights)[-n_peaks:]
    top_peaks = peaks[top_indices]
    
    # Sort by position (left to right)
    top_peaks = np.sort(top_peaks)
    
    return top_peaks


def validate_peak_quality(x_projection, peaks, min_peak_height=10, min_peak_width=20):
    """Check if detected peaks are likely to be real shoots versus noise.
    
    Validates peak quality by checking both height (signal strength) and width
    (spatial extent). Narrow spikes indicate noise, while wide peaks indicate
    actual shoots. Requires at least 3 peaks to pass validation.
    
    Args:
        x_projection: 1D array of pixel counts.
        peaks: Array of peak X-coordinates from find_shoot_peaks.
        min_peak_height: Minimum height for a valid peak (default: 10).
        min_peak_width: Minimum width at half-height for valid peak (default: 20).
        
    Returns:
        Boolean indicating if peaks are high quality (True) or likely noise (False).
        
    Example:
        >>> peaks = find_shoot_peaks(x_proj, n_peaks=5)
        >>> if validate_peak_quality(x_proj, peaks):
        ...     print("High quality peaks detected")
        ... else:
        ...     print("Peaks may be noise, consider widening Y-zone")
    """
    peak_heights = x_projection[peaks]
    
    # Calculate peak widths at half prominence
    widths, _, _, _ = peak_widths(x_projection, peaks, rel_height=0.5)
    
    # Check if at least 3 peaks are both tall enough AND wide enough
    valid_peaks = np.sum((peak_heights >= min_peak_height) & (widths >= min_peak_width))
    
    return valid_peaks >= 3


def find_shoot_peaks_with_size_fallback(mask, global_stats, n_peaks=5, min_distance=300,
                                        x_min=1000, initial_std=2.0, max_std=3.5,
                                        std_step=0.25, min_area=500, min_peak_width=20,
                                        quality_check_threshold=2.5):
    """Adaptively find shoot peaks with fallback to size-based detection.
    
    Three-stage detection strategy:
    1. Normal cases (std 2.0-2.25): X-projection peaks without quality checks
    2. Widened search (std 2.5-3.5): Progressive Y-zone widening with quality validation
    3. Size fallback: Largest components by area when projection methods fail
    
    This handles both typical shoots and edge cases like failed germination or
    shoots that have fallen outside the typical vertical zone.
    
    Args:
        mask: Binary mask with values 0 and 255.
        global_stats: Global statistics from calculate_global_y_stats.
        n_peaks: Target number of peaks (default: 5).
        min_distance: Minimum distance between peaks in pixels (default: 300).
        x_min: Minimum X coordinate to consider (default: 1000).
        initial_std: Starting std multiplier (default: 2.0).
        max_std: Maximum std multiplier to try (default: 3.5).
        std_step: Step size for widening (default: 0.25).
        min_area: Minimum area for size fallback in pixels (default: 500).
        min_peak_width: Minimum peak width for quality validation (default: 20).
        quality_check_threshold: Std threshold to activate quality checks (default: 2.5).
        
    Returns:
        Tuple of (peaks, x_projection, std_multiplier_used, method_used) where:
            - peaks: Array of peak X-coordinates
            - x_projection: The X-projection array used
            - std_multiplier_used: The std multiplier that succeeded
            - method_used: Either "projection" or "size_fallback"
        
    Example:
        >>> peaks, x_proj, std, method = find_shoot_peaks_with_size_fallback(
        ...     mask, global_stats, quality_check_threshold=2.5
        ... )
        >>> print(f"Method: {method}, Std: {std:.2f}")
        >>> print(f"Peaks at: {peaks}")
    """
    std_multiplier = initial_std
    
    # Try standard X-projection approach with progressive widening
    while std_multiplier <= max_std:
        x_projection = calculate_x_projection(mask, global_stats, std_multiplier)
        peaks = find_shoot_peaks(x_projection, n_peaks, min_distance, x_min)
        
        if len(peaks) >= n_peaks:
            # Only check quality if we're in desperate territory (high std)
            if std_multiplier >= quality_check_threshold:
                if validate_peak_quality(x_projection, peaks, min_peak_height=10, 
                                       min_peak_width=min_peak_width):
                    print(f"Found {len(peaks)} quality peaks with std_multiplier={std_multiplier:.2f}")
                    return peaks, x_projection, std_multiplier, "projection"
                else:
                    print(f"Found {len(peaks)} peaks but quality too low (std={std_multiplier:.2f})")
            else:
                # Low std values - just trust the peaks
                print(f"Found {len(peaks)} peaks with std_multiplier={std_multiplier:.2f}")
                return peaks, x_projection, std_multiplier, "projection"
        
        std_multiplier += std_step
    
    # Fallback: Use connected components filtered by size and X position
    print(f"Projection method insufficient, trying size-based fallback...")
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x_left = stats[i, cv2.CC_STAT_LEFT]
        width = stats[i, cv2.CC_STAT_WIDTH]
        bbox_center = x_left + width // 2
        
        if area >= min_area and bbox_center >= x_min:
            valid_components.append({
                'label': i,
                'bbox_center': bbox_center,
                'area': area
            })
    
    # Sort by area (largest first) and take top n_peaks
    valid_components.sort(key=lambda x: x['area'], reverse=True)
    selected = valid_components[:n_peaks]
    
    # Extract X positions and sort left to right
    peaks = np.array([c['bbox_center'] for c in selected])
    peaks = np.sort(peaks)
    
    # Create full projection for visualization
    x_projection = np.sum(mask > 0, axis=0)
    
    print(f"Found {len(peaks)} peaks using size-based fallback (min_area={min_area})")
    return peaks, x_projection, max_std, "size_fallback"


def merge_with_narrow_bands(mask, peaks, global_stats, std_multiplier, band_width=100):
    """Find components near peaks and keep entire components (not clipped to bands).
    
    Uses narrow X-bands around detected peaks to identify which components belong
    to each shoot location. Keeps complete components rather than clipping them
    to band boundaries, preserving shoot morphology.
    
    Args:
        mask: Binary mask with values 0 and 255.
        peaks: Array of peak X-coordinates from find_shoot_peaks_with_size_fallback.
        global_stats: Global statistics from calculate_global_y_stats.
        std_multiplier: Std multiplier used for peak detection.
        band_width: Half-width of X-band for identifying components (default: 100).
        
    Returns:
        Binary mask with complete components near peaks (values 0 and 255).
        
    Example:
        >>> peaks, x_proj, std, method = find_shoot_peaks_with_size_fallback(
        ...     mask, global_stats
        ... )
        >>> filtered = merge_with_narrow_bands(mask, peaks, global_stats, std, 
        ...                                    band_width=100)
    """
    # Find all connected components in the full mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Create empty output
    output_mask = np.zeros_like(mask)
    
    # For each peak, find components whose center falls in the band
    kept_labels = set()
    
    for i, peak_x in enumerate(peaks):
        x_left = peak_x - band_width
        x_right = peak_x + band_width
        
        # Check each component
        for label in range(1, num_labels):
            if label in kept_labels:
                continue  # Already assigned to another peak
            
            # Get component's X-center
            comp_x_left = stats[label, cv2.CC_STAT_LEFT]
            comp_width = stats[label, cv2.CC_STAT_WIDTH]
            comp_x_center = comp_x_left + comp_width // 2
            
            # If center is in this peak's band, keep the ENTIRE component
            if x_left <= comp_x_center <= x_right:
                output_mask[labels == label] = 255
                kept_labels.add(label)
                print(f"  Peak {i+1} at X={peak_x}: keeping component with center at X={comp_x_center}")
    
    return output_mask


def filter_one_per_peak(mask, peaks, global_stats, std_multiplier, band_width=100,
                       require_y_zone=True):
    """Keep the largest component near each peak, optionally validating Y-zone.
    
    For each detected peak, identifies all components within the X-band and keeps
    only the largest one. Optionally filters out components that don't touch the
    Y-zone (disabled for size-fallback method to handle fallen shoots).
    
    Args:
        mask: Binary mask with values 0 and 255.
        peaks: Array of peak X-coordinates.
        global_stats: Global statistics from calculate_global_y_stats.
        std_multiplier: Std multiplier used for peak detection.
        band_width: Half-width for assigning components to peaks (default: 100).
        require_y_zone: If True, only keep components touching Y-zone (default: True).
        
    Returns:
        Binary mask with exactly one component per peak (values 0 and 255).
        
    Example:
        >>> # For projection method (require Y-zone)
        >>> final_mask = filter_one_per_peak(mask, peaks, global_stats, std_used,
        ...                                  band_width=100, require_y_zone=True)
        >>> 
        >>> # For size fallback (allow fallen shoots outside Y-zone)
        >>> final_mask = filter_one_per_peak(mask, peaks, global_stats, std_used,
        ...                                  band_width=100, require_y_zone=False)
    
    Note:
        Set require_y_zone=False when using size-based fallback method to preserve
        large shoots that have fallen outside the typical vertical zone.
    """
    # Calculate Y-zone bounds
    global_upper = int(global_stats['global_mean'] - std_multiplier * global_stats['global_std'])
    global_lower = int(global_stats['global_mean'] + std_multiplier * global_stats['global_std'])
    
    # Find all components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    output_mask = np.zeros_like(mask)
    
    for peak_x in peaks:
        x_left = peak_x - band_width
        x_right = peak_x + band_width
        
        # Find all components in this peak's band
        candidates = []
        for i in range(1, num_labels):
            x_center = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
            
            if x_left <= x_center <= x_right:
                area = stats[i, cv2.CC_STAT_AREA]
                
                if require_y_zone:
                    # Check if component touches Y-zone
                    component_mask = (labels == i).astype(np.uint8)
                    y_zone_pixels = np.sum(component_mask[global_upper:global_lower, :])
                    
                    if y_zone_pixels > 0:
                        candidates.append((i, area))
                else:
                    # No Y-zone requirement - keep all
                    candidates.append((i, area))
        
        # Keep the largest one from this peak
        if candidates:
            best_label = max(candidates, key=lambda x: x[1])[0]
            output_mask[labels == best_label] = 255
            print(f"  Peak at X={peak_x}: keeping component {best_label} "
                  f"(area={stats[best_label, cv2.CC_STAT_AREA]})")
        else:
            print(f"  Peak at X={peak_x}: WARNING - no valid components found!")
    
    return output_mask


def clean_shoot_mask_pipeline(mask_path, global_stats, closing_kernel_size=7,
                              closing_iterations=5, quality_check_threshold=2.5,
                              band_width=100):
    """Complete pipeline to clean a shoot mask from raw predictions to final output.
    
    Runs the full processing workflow:
    1. Load mask and apply initial morphological closing
    2. Detect 5 shoot locations using adaptive peak finding
    3. Filter to components near detected peaks
    4. Keep exactly one component per peak (largest in each band)
    
    Args:
        mask_path: Path to shoot mask file (string or Path object).
        global_stats: Global statistics from calculate_global_y_stats.
        closing_kernel_size: Initial closing kernel size (default: 7).
        closing_iterations: Initial closing iterations (default: 5).
        quality_check_threshold: Std threshold for quality checks (default: 2.5).
        band_width: X-band width around peaks in pixels (default: 100).
        
    Returns:
        Dictionary with keys:
            - 'cleaned_mask': Final cleaned binary mask (0 and 255)
            - 'peaks': Detected peak X-coordinates
            - 'std_used': Std multiplier that succeeded
            - 'method': Detection method ("projection" or "size_fallback")
            - 'num_components': Number of components in final mask
            - 'filename': Input filename
        
    Example:
        >>> from pathlib import Path
        >>> 
        >>> # Process all masks
        >>> mask_dir = Path('data/shoot_masks')
        >>> for mask_file in mask_dir.glob('*.png'):
        ...     result = clean_shoot_mask_pipeline(str(mask_file), global_stats)
        ...     
        ...     # Save cleaned mask
        ...     output_path = f"cleaned/{result['filename']}"
        ...     cv2.imwrite(output_path, result['cleaned_mask'])
        ...     
        ...     # Log results
        ...     if result['num_components'] != 5:
        ...         print(f"WARNING: {result['filename']} has "
        ...               f"{result['num_components']} components")
    
    Note:
        Parameters are optimized for typical plant imaging conditions:
        - Shoots at X > 1000 pixels
        - Shoots between Y = 200-750 pixels (adaptive per image)
        - ~400-500 pixel spacing between plants
    """
    print(f"Processing: {Path(mask_path).name}")
    
    # Step 1: Load and initial joining
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    joined_mask = join_shoot_fragments(mask, closing_kernel_size, closing_iterations)
    
    # Step 2: Find shoot locations
    peaks, x_projection, std_used, method = find_shoot_peaks_with_size_fallback(
        joined_mask, global_stats,
        quality_check_threshold=quality_check_threshold
    )
    print(f"  Found {len(peaks)} peaks using {method} (std={std_used:.2f})")
    
    # Step 3: Filter to narrow bands around peaks
    narrow_result = merge_with_narrow_bands(joined_mask, peaks, global_stats, 
                                           std_used, band_width=band_width)
    
    # Step 4: Keep one component per peak
    require_y_zone = (method != "size_fallback")
    final_mask = filter_one_per_peak(narrow_result, peaks, global_stats, std_used,
                                    band_width=band_width, require_y_zone=require_y_zone)
    
    # Count final components
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    num_components = num_labels - 1
    
    return {
        'cleaned_mask': final_mask,
        'peaks': peaks,
        'std_used': std_used,
        'method': method,
        'num_components': num_components,
        'filename': Path(mask_path).name
    }


# Recommended parameters as a reference
RECOMMENDED_PARAMS = {
    'initial_closing': {
        'kernel_size': 7,
        'iterations': 5
    },
    'y_zone_boundaries': {
        'y_min': 200,  # Shoots never above this
        'y_max': 750   # Shoots never below this
    },
    'peak_detection': {
        'n_peaks': 5,
        'min_distance': 300,          # Minimum spacing between shoots
        'x_min': 1000,                # Shoots never left of this
        'initial_std': 2.0,           # Starting Y-zone width
        'max_std': 3.5,               # Maximum Y-zone width
        'quality_check_threshold': 2.5,  # When to validate peaks
        'min_peak_width': 20,         # Minimum width for valid peaks
        'min_peak_height': 10,        # Minimum height for valid peaks
        'min_area': 500               # For size-based fallback
    },
    'filtering': {
        'band_width': 100  # X-distance around peaks
    }
}
