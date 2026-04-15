from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
import skimage.measure




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

def repair_root_mask_edges(mask_path, 
                          min_component_size=20,
                          max_edge_distance=50,
                          max_iterations=10,
                          closing_kernel_height=5,
                          closing_kernel_width=3,
                          closing_iterations=2,
                          save_path=None,
                          visualize=False):
    """Repair root mask gaps using edge-to-edge distance measurements.
    
    Uses iterative edge-based gap filling to connect fragmented root components.
    Measures minimum distance between component edges rather than centroids,
    which is more accurate for irregular shapes like roots.
    
    Args:
        mask_path: Path to root mask file (str or Path).
        min_component_size: Minimum component area in pixels to consider for connection.
        max_edge_distance: Maximum pixel distance between edges to connect components.
        max_iterations: Maximum number of gap-filling iterations.
        closing_kernel_height: Height of initial morphological closing kernel.
        closing_kernel_width: Width of initial morphological closing kernel.
        closing_iterations: Number of initial closing iterations.
        save_path: Optional path to save repaired mask. If None, does not save.
        visualize: If True, displays before/after visualization with original image.
        
    Returns:
        Binary mask array (0 and 255) with repaired gaps.
        
    Example:
        repaired_mask = repair_root_mask_edges(
            'root_mask.png',
            max_edge_distance=50,
            visualize=True
        )
    """
    
    # Load mask
    mask = load_mask(mask_path)
    
    # Initial morphological closing
    kernel = np.ones((closing_kernel_height, closing_kernel_width), np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 
                                   iterations=closing_iterations)
    
    repaired_mask = closed_mask.copy()
    
    # Iterative edge-based gap filling
    for iteration in range(max_iterations):
        retval, labels = cv2.connectedComponents(repaired_mask)
        regions = skimage.measure.regionprops(labels)
        
        # Get substantial components with their edge pixels
        components = []
        for region in regions:
            if region.area < min_component_size:
                continue
            
            coords = region.coords
            min_y = coords[:, 0].min()
            max_y = coords[:, 0].max()
            
            # Get bottom and top edge pixels
            bottom_edge = coords[coords[:, 0] == max_y]
            top_edge = coords[coords[:, 0] == min_y]
            
            components.append({
                'label': region.label,
                'area': region.area,
                'min_y': min_y,
                'max_y': max_y,
                'bottom_edge': bottom_edge,
                'top_edge': top_edge
            })
        
        connections_made = 0
        
        # Find and connect close component edges
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components):
                if i >= j:
                    continue
                
                # Check if comp2 is below comp1
                if comp1['max_y'] >= comp2['min_y']:
                    continue
                
                # Calculate minimum distance between edges
                distances = cdist(comp1['bottom_edge'], comp2['top_edge'])
                min_distance = distances.min()
                min_idx = np.unravel_index(distances.argmin(), distances.shape)
                
                # Get the closest points
                closest_bottom = comp1['bottom_edge'][min_idx[0]]
                closest_top = comp2['top_edge'][min_idx[1]]
                
                if min_distance <= max_edge_distance:
                    # Draw line between closest points
                    cv2.line(repaired_mask, 
                            (closest_bottom[1], closest_bottom[0]),
                            (closest_top[1], closest_top[0]), 
                            255, 3)
                    connections_made += 1
        
        # Apply closing to smooth connections
        kernel_smooth = np.ones((5, 3), np.uint8)
        repaired_mask = cv2.morphologyEx(repaired_mask, cv2.MORPH_CLOSE, 
                                        kernel_smooth, iterations=1)
        
        # Stop if no connections were made
        if connections_made == 0:
            break
    
    # Save if path provided
    if save_path is not None:
        cv2.imwrite(str(save_path), repaired_mask)
    
    # Visualize if requested
    if visualize:
        mask_path_obj = Path(mask_path)
        mask_name = mask_path_obj.stem
        image_num = mask_name.replace('test_image_', '').replace('_root', '')
        
        # Try to find original image
        images_path = mask_path_obj.parent.parent.parent / 'Kaggle'
        original_path = images_path / f'test_image_{image_num}.png'
        
        retval_before = cv2.connectedComponents(closed_mask)[0]
        retval_after, labels_after = cv2.connectedComponents(repaired_mask)
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 30))
        
        axes[0].imshow(labels_after, cmap='gist_ncar')
        axes[0].set_title(f'After Edge-based Repair: {retval_after - 1} components\n'
                         f'(started with {retval_before - 1})', fontsize=16)
        axes[0].axis('off')
        
        if original_path.exists():
            original = cv2.imread(str(original_path))
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            axes[1].imshow(original_rgb)
            axes[1].imshow(labels_after, cmap='gist_ncar', alpha=0.4)
            axes[1].set_title('Overlay with Original Image', fontsize=16)
        else:
            axes[1].imshow(labels_after, cmap='gist_ncar')
            axes[1].set_title('Repaired Mask (no original image found)', fontsize=16)
        
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return repaired_mask