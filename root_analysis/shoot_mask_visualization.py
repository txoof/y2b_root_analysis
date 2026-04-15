"""Visualization and debugging tools for shoot mask cleaning pipeline.

This module provides visualization functions for analyzing shoot mask cleaning
results, debugging pipeline failures, and validating parameter choices. Works in
conjunction with shoot_mask_cleaning.py.

Typical usage:

    Visualize Y-density statistics:
    
        >>> import matplotlib.pyplot as plt
        >>> from shoot_mask_cleaning import (
        ...     calculate_y_density, 
        ...     calculate_weighted_y_stats,
        ...     calculate_global_y_stats,
        ...     join_shoot_fragments
        ... )
        >>> from shoot_mask_visualization import visualize_y_density_with_global_stats
        >>> 
        >>> # Calculate global stats
        >>> mask_files = [str(f) for f in Path('data/shoot').glob('*.png')]
        >>> global_stats = calculate_global_y_stats(mask_files)
        >>> 
        >>> # Visualize individual image
        >>> mask = cv2.imread('data/shoot/image_01.png', cv2.IMREAD_GRAYSCALE)
        >>> joined = join_shoot_fragments(mask)
        >>> density = calculate_y_density(joined)
        >>> local_stats = calculate_weighted_y_stats(density, y_min=200, y_max=750)
        >>> 
        >>> visualize_y_density_with_global_stats(joined, density, local_stats, 
        ...                                       global_stats, std_multiplier=2.0)
        >>> plt.show()
    
    Debug complete pipeline:
    
        >>> from shoot_mask_visualization import debug_peak_detection
        >>> 
        >>> # Visualize all processing steps for troubleshooting
        >>> debug_peak_detection('data/shoot/problematic_image.png', global_stats)
        >>> plt.show()
    
    Visualize final results:
    
        >>> from shoot_mask_cleaning import clean_shoot_mask_pipeline
        >>> from shoot_mask_visualization import visualize_final_components
        >>> 
        >>> result = clean_shoot_mask_pipeline('mask.png', global_stats)
        >>> visualize_final_components(result['cleaned_mask'], 
        ...                            title=f"Final: {result['filename']}")
        >>> plt.show()

Dependencies:
    - matplotlib
    - numpy
    - opencv-python (cv2)
    - scipy
    - shoot_mask_cleaning (companion module)

Author: Aaron Ciuffo
Date: December 2024
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_y_density_with_global_stats(mask, density, local_stats, global_stats,
                                          std_multiplier=2.0, figsize=(10, 8)):
    """Visualize mask with both local and global Y-density statistics.
    
    Creates a two-panel figure showing:
    1. Y-density histogram with local and global zones highlighted
    2. Mask with corresponding zone overlays and mean position markers
    
    Args:
        mask: Binary mask array (0 and 255).
        density: 1D array of pixel counts per row from calculate_y_density.
        local_stats: Local statistics dict with 'mean' and 'std' keys.
        global_stats: Global statistics from calculate_global_y_stats.
        std_multiplier: Standard deviation multiplier for zone width (default: 2.0).
        figsize: Figure size as (width, height) tuple (default: (10, 8)).
        
    Example:
        >>> from shoot_mask_cleaning import (
        ...     calculate_y_density,
        ...     calculate_weighted_y_stats,
        ...     join_shoot_fragments
        ... )
        >>> 
        >>> mask = cv2.imread('shoot.png', cv2.IMREAD_GRAYSCALE)
        >>> joined = join_shoot_fragments(mask)
        >>> density = calculate_y_density(joined)
        >>> local_stats = calculate_weighted_y_stats(density, y_min=200, y_max=750)
        >>> 
        >>> visualize_y_density_with_global_stats(joined, density, local_stats,
        ...                                       global_stats, std_multiplier=2.0)
        >>> plt.show()
    
    Note:
        Red = local image statistics
        Yellow = global dataset statistics
        Orange markers = mean positions
    """
    fig, (ax_hist, ax_mask) = plt.subplots(1, 2, figsize=figsize,
                                            gridspec_kw={'width_ratios': [1, 3]})
    
    # Plot density histogram
    ax_hist.barh(range(len(density)), density, height=1, color='blue', alpha=0.6)
    ax_hist.set_ylim(len(density), 0)
    ax_hist.set_xlabel('Pixel count')
    ax_hist.set_ylabel('Y coordinate')
    ax_hist.invert_xaxis()
    
    # Calculate bounds
    local_upper = local_stats['mean'] - std_multiplier * local_stats['std']
    local_lower = local_stats['mean'] + std_multiplier * local_stats['std']
    global_upper = global_stats['global_mean'] - std_multiplier * global_stats['global_std']
    global_lower = global_stats['global_mean'] + std_multiplier * global_stats['global_std']
    
    # Draw ROI bounds (gray)
    if 'y_min' in global_stats and 'y_max' in global_stats:
        ax_hist.axhline(global_stats['y_min'], color='gray', linestyle='-',
                       linewidth=1, alpha=0.5)
        ax_hist.axhline(global_stats['y_max'], color='gray', linestyle='-',
                       linewidth=1, alpha=0.5)
    
    # Draw shaded zones on histogram
    ax_hist.axhspan(local_upper, local_lower, alpha=0.2, color='red', label='Local zone')
    ax_hist.axhspan(global_upper, global_lower, alpha=0.2, color='yellow', label='Global zone')
    
    # Draw mean lines
    ax_hist.axhline(local_stats['mean'], color='red', linestyle='-',
                   linewidth=2, label='Local mean')
    ax_hist.axhline(global_stats['global_mean'], color='orange', linestyle='-',
                   linewidth=2, label='Global mean')
    
    ax_hist.legend(loc='upper right', fontsize=8)
    ax_hist.grid(True, alpha=0.3)
    
    # Show mask with overlays
    ax_mask.imshow(mask, cmap='gray', aspect='equal')
    
    # Draw ROI bounds
    if 'y_min' in global_stats and 'y_max' in global_stats:
        ax_mask.axhline(global_stats['y_min'], color='gray', linestyle='-',
                       linewidth=1, alpha=0.5)
        ax_mask.axhline(global_stats['y_max'], color='gray', linestyle='-',
                       linewidth=1, alpha=0.5)
    
    # Draw shaded zones on mask
    ax_mask.axhspan(local_upper, local_lower, alpha=0.15, color='red')
    ax_mask.axhspan(global_upper, global_lower, alpha=0.2, color='yellow')
    
    # Add left edge markers for means (points only)
    ax_mask.plot(0, local_stats['mean'], 'o', color='red', markersize=10)
    ax_mask.plot(0, global_stats['global_mean'], 'o', color='orange', markersize=10)
    
    ax_mask.axis('off')
    
    plt.tight_layout()


def visualize_x_projection_with_peaks_adaptive(mask, x_projection, peaks, global_stats,
                                                std_multiplier, figsize=(14, 8)):
    """Visualize mask and X-projection with detected peak locations.
    
    Creates a two-panel figure showing:
    1. Mask with Y-zone overlay and vertical lines at peak positions
    2. X-projection histogram with peaks marked
    
    Args:
        mask: Binary mask array (0 and 255).
        x_projection: 1D array of pixel counts from calculate_x_projection.
        peaks: Array of peak X-coordinates.
        global_stats: Global statistics from calculate_global_y_stats.
        std_multiplier: The std multiplier actually used for this image.
        figsize: Figure size as (width, height) tuple (default: (14, 8)).
        
    Example:
        >>> from shoot_mask_cleaning import (
        ...     join_shoot_fragments,
        ...     find_shoot_peaks_with_size_fallback
        ... )
        >>> 
        >>> mask = cv2.imread('shoot.png', cv2.IMREAD_GRAYSCALE)
        >>> joined = join_shoot_fragments(mask)
        >>> peaks, x_proj, std, method = find_shoot_peaks_with_size_fallback(
        ...     joined, global_stats
        ... )
        >>> 
        >>> visualize_x_projection_with_peaks_adaptive(joined, x_proj, peaks,
        ...                                            global_stats, std)
        >>> plt.show()
    """
    fig, (ax_mask, ax_proj) = plt.subplots(2, 1, figsize=figsize,
                                            gridspec_kw={'height_ratios': [3, 1]})
    
    # Calculate Y bounds
    global_upper = int(global_stats['global_mean'] - std_multiplier * global_stats['global_std'])
    global_lower = int(global_stats['global_mean'] + std_multiplier * global_stats['global_std'])
    
    # Show mask with peaks
    ax_mask.imshow(mask, cmap='gray', aspect='equal')
    ax_mask.axhspan(global_upper, global_lower, alpha=0.2, color='yellow')
    
    for i, peak_x in enumerate(peaks):
        ax_mask.axvline(peak_x, color='red', linestyle='--', linewidth=2)
        ax_mask.text(peak_x, 100, f'{i+1}', color='red', fontsize=12,
                    ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_mask.set_title(f'Mask with {len(peaks)} Detected Peaks (std={std_multiplier:.2f})')
    ax_mask.axis('off')
    
    # Show X-projection with peaks
    ax_proj.fill_between(range(len(x_projection)), x_projection, alpha=0.7, color='blue')
    ax_proj.set_xlim(0, len(x_projection))
    ax_proj.set_xlabel('X coordinate (pixels)')
    ax_proj.set_ylabel('Pixel count')
    ax_proj.set_title('X-axis Projection')
    ax_proj.grid(True, alpha=0.3)
    
    # Mark peaks
    for peak_x in peaks:
        ax_proj.axvline(peak_x, color='red', linestyle='--', linewidth=2)
        ax_proj.plot(peak_x, x_projection[peak_x], 'ro', markersize=10)
    
    plt.tight_layout()


def visualize_final_components(mask, global_stats, std_used, title="Final Mask", 
                               figsize=(16, 6)):
    """Visualize final mask with component count, labels, and statistics.
    
    Creates a two-panel figure showing:
    1. Binary mask with component count
    2. Colored label map with component numbers positioned below each shoot
    
    Args:
        mask: Binary mask array (0 and 255).
        global_stats: Global statistics from calculate_global_y_stats.
        std_used: Std multiplier used for this image.
        title: Title prefix for the plot (default: "Final Mask").
        figsize: Figure size as (width, height) tuple (default: (16, 6)).
        
    Returns:
        Number of components found in the mask.
        
    Example:
        >>> from shoot_mask_cleaning import clean_shoot_mask_pipeline
        >>> 
        >>> result = clean_shoot_mask_pipeline('mask.png', global_stats)
        >>> num_components = visualize_final_components(
        ...     result['cleaned_mask'],
        ...     global_stats,
        ...     result['std_used'],
        ...     title=f"Final: {result['filename']}"
        ... )
        >>> 
        >>> if num_components != 5:
        ...     print(f"WARNING: Expected 5, found {num_components}")
        >>> plt.show()
    
    Note:
        Component labels are positioned below each shoot to avoid obscuring
        the actual shoot structures. Labeled view is zoomed to the Y-zone area
        for easier inspection.
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    num_components = num_labels - 1  # Exclude background
    
    # Create a colored label image for visualization
    label_colors = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Assign random colors to each component
    np.random.seed(42)
    for i in range(1, num_labels):
        color = np.random.randint(50, 255, size=3)
        label_colors[labels == i] = color
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Original mask
    ax1.imshow(mask, cmap='gray', aspect='equal')
    ax1.set_title(f'{title}\nComponents: {num_components}')
    ax1.axis('off')
    
    # Colored labels with centroids - zoomed to area of interest
    ax2.imshow(label_colors, aspect='equal')
    
    # Calculate Y-zone for zoom
    global_upper = int(global_stats['global_mean'] - std_used * global_stats['global_std'])
    global_lower = int(global_stats['global_mean'] + std_used * global_stats['global_std'])
    
    # Add padding
    y_padding = 100
    zoom_y_min = max(0, global_upper - y_padding)
    zoom_y_max = min(mask.shape[0], global_lower + y_padding)
    
    # Set axis limits to zoom
    ax2.set_ylim(zoom_y_max, zoom_y_min)  # Inverted for image coordinates
    ax2.set_xlim(0, mask.shape[1])
    
    # Mark centroids and label them - place labels below objects
    for i in range(1, num_labels):
        cx, cy = centroids[i]
        # Get component bottom Y coordinate
        y_bottom = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        
        # Place label below the object
        ax2.text(cx, y_bottom + 40, f'{i}', color='white', fontsize=14,
                ha='center', weight='bold',
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax2.set_title(f'Labeled Components: {num_components} (Zoomed)')
    ax2.axis('off')
    
    plt.tight_layout()
    
    # Print component stats
    print(f"\nComponent Statistics:")
    print(f"{'Label':<8} {'Area':<10} {'X-Center':<10} {'Y-Center':<10}")
    print("-" * 40)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cx, cy = centroids[i]
        print(f"{i:<8} {area:<10} {cx:<10.1f} {cy:<10.1f}")
    
    return num_components


def debug_peak_detection(mask_path, global_stats, kernel_size=7, iterations=5,
                        quality_check_threshold=2.5):
    """Debug visualization showing all pipeline steps with peak locations.
    
    Creates a comprehensive four-panel figure for troubleshooting:
    1. Original mask with detected peak locations and Y-zone
    2. Mask after morphological closing with peaks
    3. X-projection histogram with marked peaks
    4. Final cleaned result with component labels
    
    Args:
        mask_path: Path to shoot mask file (string or Path object).
        global_stats: Global statistics from calculate_global_y_stats.
        kernel_size: Kernel size for initial closing (default: 7).
        iterations: Number of closing iterations (default: 5).
        quality_check_threshold: Std threshold for quality checks (default: 2.5).
        
    Example:
        >>> # Debug a problematic image
        >>> debug_peak_detection('data/shoot/image_11.png', global_stats)
        >>> plt.savefig('debug_output.png', dpi=150, bbox_inches='tight')
        >>> plt.show()
        >>> 
        >>> # Batch debug multiple images
        >>> for mask_file in problematic_files:
        ...     debug_peak_detection(mask_file, global_stats)
        ...     plt.show()
    
    Note:
        This function prints detailed information about peak detection results,
        component assignments, and filtering decisions to help diagnose issues.
    """
    from library.shoot_mask_cleaning import (
        join_shoot_fragments,
        find_shoot_peaks_with_size_fallback,
        merge_with_narrow_bands,
        filter_one_per_peak
    )
    
    print(f"Debugging: {Path(mask_path).name}")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    joined_mask = join_shoot_fragments(mask, kernel_size=kernel_size, iterations=iterations)
    
    # Find peaks
    peaks, x_projection, std_used, method = find_shoot_peaks_with_size_fallback(
        joined_mask, global_stats, quality_check_threshold=quality_check_threshold
    )
    
    print(f"  Peaks detected at X positions: {peaks}")
    print(f"  Method: {method}, Std: {std_used:.2f}")
    
    # Process
    narrow_result = merge_with_narrow_bands(joined_mask, peaks, global_stats, 
                                           std_used, band_width=100)
    
    require_y_zone = (method != "size_fallback")
    final_result = filter_one_per_peak(narrow_result, peaks, global_stats, std_used,
                                      band_width=100, require_y_zone=require_y_zone)
    
    # Calculate Y-zone
    global_upper = int(global_stats['global_mean'] - std_used * global_stats['global_std'])
    global_lower = int(global_stats['global_mean'] + std_used * global_stats['global_std'])
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Original with peaks
    axes[0, 0].imshow(mask, cmap='gray', aspect='equal')
    axes[0, 0].axhspan(global_upper, global_lower, alpha=0.2, color='yellow')
    for i, peak_x in enumerate(peaks):
        axes[0, 0].axvline(peak_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
        axes[0, 0].text(peak_x, 50, f'P{i+1}', color='red', fontsize=12,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    axes[0, 0].set_title('Original with Peak Locations')
    axes[0, 0].axis('off')
    
    # After joining with peaks
    axes[0, 1].imshow(joined_mask, cmap='gray', aspect='equal')
    axes[0, 1].axhspan(global_upper, global_lower, alpha=0.2, color='yellow')
    for i, peak_x in enumerate(peaks):
        axes[0, 1].axvline(peak_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].set_title('After Closing with Peaks')
    axes[0, 1].axis('off')
    
    # X-projection
    axes[1, 0].fill_between(range(len(x_projection)), x_projection, alpha=0.7, color='blue')
    axes[1, 0].set_xlim(0, len(x_projection))
    for i, peak_x in enumerate(peaks):
        axes[1, 0].axvline(peak_x, color='red', linestyle='--', linewidth=2)
        axes[1, 0].plot(peak_x, x_projection[peak_x], 'ro', markersize=10)
        axes[1, 0].text(peak_x, x_projection[peak_x] + 5, f'P{i+1}',
                       ha='center', color='red', weight='bold')
    axes[1, 0].set_xlabel('X coordinate')
    axes[1, 0].set_ylabel('Pixel count')
    axes[1, 0].set_title('X-Projection with Detected Peaks')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final with component labels
    num_labels, labels_map, comp_stats, comp_centroids = cv2.connectedComponentsWithStats(
        final_result, connectivity=8
    )
    axes[1, 1].imshow(final_result, cmap='gray', aspect='equal')
    axes[1, 1].axhspan(global_upper, global_lower, alpha=0.2, color='yellow')
    for i in range(1, num_labels):
        cx, cy = comp_centroids[i]
        y_bottom = comp_stats[i, cv2.CC_STAT_TOP] + comp_stats[i, cv2.CC_STAT_HEIGHT]
        axes[1, 1].text(cx, y_bottom + 40, f'{i}', color='red', fontsize=14,
                       ha='center', weight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    axes[1, 1].set_title(f'Final: {num_labels-1} components')
    axes[1, 1].axis('off')
    
    plt.tight_layout()


def batch_process_and_visualize(mask_paths, global_stats, output_dir=None,
                                save_masks=True, show_plots=True):
    """Process all masks and generate summary visualizations.
    
    Runs the complete cleaning pipeline on multiple masks and creates:
    1. Individual processed masks (saved to output_dir if specified)
    2. Summary statistics printed to console
    3. Optional visualization of each result
    
    Args:
        mask_paths: List of paths to mask files.
        global_stats: Global statistics from calculate_global_y_stats.
        output_dir: Directory to save cleaned masks (default: None = don't save).
        save_masks: Whether to save cleaned masks (default: True).
        show_plots: Whether to display plots for each image (default: True).
        
    Returns:
        List of result dictionaries from clean_shoot_mask_pipeline.
        
    Example:
        >>> from pathlib import Path
        >>> 
        >>> mask_files = [str(f) for f in Path('data/shoot').glob('*.png')]
        >>> results = batch_process_and_visualize(
        ...     mask_files,
        ...     global_stats,
        ...     output_dir='data/shoot_cleaned',
        ...     save_masks=True,
        ...     show_plots=False
        ... )
        >>> 
        >>> # Summary
        >>> success_count = sum(1 for r in results if r['num_components'] == 5)
        >>> print(f"Success rate: {success_count}/{len(results)}")
    """
    from shoot_mask_cleaning import clean_shoot_mask_pipeline
    
    if output_dir is not None and save_masks:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for mask_path in mask_paths:
        # Process mask
        result = clean_shoot_mask_pipeline(mask_path, global_stats)
        results.append(result)
        
        # Save if requested
        if output_dir is not None and save_masks:
            save_path = output_path / result['filename']
            cv2.imwrite(str(save_path), result['cleaned_mask'])
        
        # Visualize if requested
        if show_plots:
            visualize_final_components(result['cleaned_mask'], global_stats,
                                      result['std_used'],
                                      title=f"Final: {result['filename']}")
            plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"{'Filename':<30} {'Method':<15} {'Std':>6} {'Components':>12}")
    print("-"*70)
    
    for result in results:
        print(f"{result['filename']:<30} {result['method']:<15} "
              f"{result['std_used']:>6.2f} {result['num_components']:>12}")
    
    # Summary statistics
    total = len(results)
    correct_count = sum(1 for r in results if r['num_components'] == 5)
    projection_count = sum(1 for r in results if r['method'] == 'projection')
    
    print("-"*70)
    print(f"Total processed: {total}")
    print(f"Correct component count (5): {correct_count} ({100*correct_count/total:.1f}%)")
    print(f"Projection method: {projection_count} ({100*projection_count/total:.1f}%)")
    print(f"Size fallback method: {total-projection_count} ({100*(total-projection_count)/total:.1f}%)")
    print("="*70 + "\n")
    
    return results
