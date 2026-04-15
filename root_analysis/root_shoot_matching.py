"""Root-shoot matching module for plant root analysis.

This module provides a complete pipeline for matching root skeleton structures to shoot
regions in plant microscopy images. It handles filtering, scoring, assignment of roots
to shoots, and produces length measurements with robot coordinates for each plant.

The pipeline processes binary masks of shoots and roots through multiple stages:
1. Shoot reference point extraction
2. Root mask filtering by spatial sampling
3. Root structure extraction and skeletonization
4. Candidate scoring based on proximity, size, and verticality
5. Greedy left-to-right assignment ensuring one root per shoot
6. Top node identification for path tracing
7. Length calculation and coordinate conversion

Complete End-to-End Example:
-----------------------------
```
# Setup paths and create data loader
from pathlib import Path
from root_shoot_matching import GetIm, RootMatchingConfig, match_roots_to_shoots_complete
import pandas as pd

# Define paths to your masks and images
images_path = Path('data/Kaggle/')
roots_path = Path('data/repaired/roots')
shoots_path = Path('data/repaired/shoots')

# Get sorted file lists
images = sorted(images_path.glob('*.png'))
roots = sorted(roots_path.glob('*.png'))
shoots = sorted(shoots_path.glob('*.png'))

# Create getter for loading masks
getter = GetIm(shoots=shoots, roots=roots)

# Configure matching parameters
config = RootMatchingConfig(
    max_horizontal_offset=200,      # Horizontal search radius (pixels)
    distance_weight=0.35,            # Proximity importance (0-1)
    size_weight=0.35,                # Root length importance (0-1)
    verticality_weight=0.3,          # Vertical orientation importance (0-1)
    max_below_shoot=100,             # Max distance below shoot to search (pixels)
    min_skeleton_pixels=6            # Minimum pixels to be valid root
)

# Process single sample
sample_idx = 0
shoot_mask, root_mask = getter(sample_idx, is_idx=True)

# Get results as numpy array (5 lengths in pixels, left to right)
lengths = match_roots_to_shoots_complete(
    shoot_mask, root_mask, images[sample_idx], config
)
print(f"Lengths: {lengths}")  # Array of 5 floats

# Get results as DataFrame with all coordinates
df = match_roots_to_shoots_complete(
    shoot_mask, root_mask, images[sample_idx], config,
    return_dataframe=True,
    sample_idx=sample_idx,
    verbose=True
)
print(df)
# Output columns:
# - plant_order: 1-5 (left to right)
# - Plant ID: test_image_1
# - Length (px): root length in pixels
# - length_px: duplicate for compatibility
# - top_node_x, top_node_y: pixel coordinates of root start
# - endpoint_x, endpoint_y: pixel coordinates of root end
# - top_node_robot_x/y/z: robot coordinates in meters
# - endpoint_robot_x/y/z: robot coordinates in meters

# Process all samples into single DataFrame
all_results = []
for idx in range(len(roots)):
    shoot_mask, root_mask = getter(idx, is_idx=True)
    df = match_roots_to_shoots_complete(
        shoot_mask, root_mask, images[idx], config,
        return_dataframe=True,
        sample_idx=idx
    )
    all_results.append(df)

# Combine and save full dataset
results_df = pd.concat(all_results, ignore_index=True)
results_df.to_csv('root_measurements_full.csv', index=False)
print(f"Processed {len(results_df)} plants from {len(roots)} images")

# Create Kaggle submission (Plant ID and Length only)
submission_df = results_df[['Plant ID', 'Length (px)']]
submission_df.to_csv('kaggle_submission.csv', index=False)
print(f"Saved Kaggle submission with {len(submission_df)} entries")
```

Key Classes and Functions:
--------------------------
GetIm: Helper class for loading and visualizing shoot/root mask pairs
RootMatchingConfig: Configuration dataclass for tuning matching parameters
match_roots_to_shoots_complete: Main pipeline function (masks in, measurements out)
pixel_to_robot_coords: Convert pixel coordinates to robot coordinates in meters

Configuration Tuning Guide:
---------------------------
max_horizontal_offset: Increase if roots drift far from shoots (default: 200-400px)
max_below_shoot: Increase if roots start far below shoots (default: 100px)
distance_weight: Increase to favor closer roots (default: 0.35)
size_weight: Increase to favor longer roots (default: 0.35)
min_skeleton_pixels: Increase to reject more noise (default: 6px)
min_score_threshold: Increase in assign_roots_to_shoots_greedy to reject weak matches (default: 0.3)
"""

import numpy as np
import pandas as pd
from pathlib import Path  # ADDED: Missing import
import matplotlib.pyplot as plt  # ADDED: Missing import
import warnings
from dataclasses import dataclass, field
from scipy import ndimage
from skimage.measure import label as label_components
from scipy.spatial.distance import cdist

from library.mask_processing import load_mask
from library.root_analysis import (
    extract_root_structures, 
    find_farthest_endpoint_path,
    calculate_skeleton_length_px
)


class GetIm():
    """Helper class for loading and displaying shoot and root mask pairs.
    
    Provides convenient access to paired shoot and root masks from sorted file lists,
    with methods for visualization. Handles both 1-indexed file numbers and 0-indexed
    array indices.
    
    Args:
        shoots (list): Sorted list of shoot mask file paths
        roots (list): Sorted list of root mask file paths (same length as shoots)
    
    Attributes:
        shoots (list): Stored shoot mask file paths
        roots (list): Stored root mask file paths
    
    Methods:
        show: Display shoot and root masks side by side
        show_overlay: Display masks overlaid in single image with grid
        __call__: Load masks directly (same as _load_masks)
    
    Examples:
        >>> from pathlib import Path
        >>> shoots = sorted(Path('data/shoots').glob('*.png'))
        >>> roots = sorted(Path('data/roots').glob('*.png'))
        >>> getter = GetIm(shoots=shoots, roots=roots)
        >>> 
        >>> # Display by file number (1-indexed)
        >>> getter.show(1)
        >>> 
        >>> # Display by array index (0-indexed)
        >>> getter.show(0, is_idx=True)
        >>> 
        >>> # Load masks directly
        >>> shoot_mask, root_mask = getter(0, is_idx=True)
    
    Notes:
        - File numbers are 1-indexed by default (file_num=1 loads first file)
        - Array indices are 0-indexed when is_idx=True (is_idx=True, file_num=0 loads first file)
        - Shoot and root lists must be the same length and correspond to matching pairs
    """    
    def __init__(self, shoots, roots):
        self.shoots = shoots
        self.roots = roots
    
    def _load_masks(self, file_num, is_idx=False, print_f=False):
        if not is_idx:
            file_num = file_num - 1
        

        shoot_f = str(self.shoots[file_num])
        root_f = str(self.roots[file_num])
        if print_f:
            print(shoot_f, '\n', root_f)
        return (load_mask(shoot_f), load_mask(root_f))

    def show(self, file_num, is_idx=False, size=(20, 8)):
        """Display shoot and root masks side by side with filenames as titles."""
        s, r = self._load_masks(file_num, is_idx, print_f=False)
        
        # Get filenames
        if not is_idx:
            file_num = file_num - 1
        
        shoot_name = Path(self.shoots[file_num]).name
        root_name = Path(self.roots[file_num]).name
        
        # Create side-by-side plot
        fig, axes = plt.subplots(1, 2, figsize=size)
        
        axes[0].imshow(s, cmap='gray')
        axes[0].set_title(shoot_name, fontsize=12)
        axes[0].axis('off')
        
        axes[1].imshow(r, cmap='gray')
        axes[1].set_title(root_name, fontsize=12)
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def show_overlay(self, file_num, is_idx=False, size=(12, 8)):
        """Display shoot (green) and root (red) masks overlaid in single image with grid."""
        s, r = self._load_masks(file_num, is_idx, print_f=False)
        
        # Get filename
        if not is_idx:
            file_num = file_num - 1
        
        shoot_name = Path(self.shoots[file_num]).name
        root_name = Path(self.roots[file_num]).name
        combined_title = f"{shoot_name} + {root_name}"
        
        # Create RGB image
        h, w = s.shape
        rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Apply shoot mask as green
        rgb_img[s > 0] = [0, 255, 0]
        
        # Apply root mask as red (on top)
        rgb_img[r > 0] = [255, 0, 0]
        
        # Display
        fig, ax = plt.subplots(figsize=size)
        ax.imshow(rgb_img)
        ax.set_title(combined_title, fontsize=12)
        
        # Add grid lines every 200 pixels
        ax.set_xticks(np.arange(0, w, 200))
        ax.set_yticks(np.arange(0, h, 200))
        ax.grid(True, color='blue', linewidth=0.5, alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    def __call__(self, file_num, is_idx=False, print_f=False):
        """
        Args:
            file_num(int): file number
            is_idx(bool): treat file_num as array index when true
            print_f(bool): print filenames when true

        Returns:
            tuple(shoot_mask, root_mask)
        """
        return self._load_masks(file_num, is_idx, print_f=print_f)

    

@dataclass
class RootMatchingConfig:
    """Configuration parameters for root-shoot matching.
    
    Attributes:
        sampling_buffer_above: Pixels above top shoot to include in sampling box
        sampling_buffer_below: Pixels below bottom shoot to include in sampling box
        distance_weight: Weight for proximity score (0-1)
        size_weight: Weight for skeleton length score (0-1)
        verticality_weight: Weight for vertical orientation score (0-1)
        max_horizontal_offset: Maximum horizontal distance from shoot (pixels)
        max_above_shoot: Allow roots to extend this many pixels above shoot bottom
        max_below_shoot: Maximum distance below shoot bottom where root can start (pixels)
        edge_exclusion_zones: List of (min_x, max_x) tuples for edge noise regions
        max_distance: Distance normalization constant (pixels)
        max_size: Size normalization constant (pixels)
        ideal_aspect: Ideal vertical:horizontal ratio for roots
    """
    sampling_buffer_above: int = 200
    sampling_buffer_below: int = 300
    
    distance_weight: float = 0.35
    size_weight: float = 0.35
    verticality_weight: float = 0.3
    
    max_horizontal_offset: int = 200
    max_above_shoot: int = 200
    max_below_shoot: int = 100
    min_skeleton_pixels: int = 6 
    edge_exclusion_zones: list = field(default_factory=lambda: [(0, 900), (3300, 4000)])
    
    max_distance: int = 100
    max_size: int = 1000
    ideal_aspect: float = 5.0



def get_shoot_reference_points_multi(shoot_mask):
    """Calculate multiple reference points for each shoot region.
    
    Args:
        shoot_mask: Binary mask array with shoot regions (H, W)
        
    Returns:
        tuple: (labeled_shoots, num_shoots, ref_points) where:
            - labeled_shoots: Array with unique label per shoot region
            - num_shoots: Number of distinct shoot regions found
            - ref_points: Dict with structure {shoot_label: {
                'centroid': (y, x),
                'bottom_center': (y, x),
                'bottom_most': (y, x),
                'bbox': (min_y, max_y, min_x, max_x)
              }}
    
    Examples:
        >>> labeled_shoots, num_shoots, ref_points = get_shoot_reference_points_multi(shoot_mask)
        >>> print(f"Found {num_shoots} shoots")
        >>> print(ref_points[1]['bottom_center'])
    """
    # Label shoots
    # labeled_shoots, num_shoots = ndimage.label(shoot_mask)
    structure = ndimage.generate_binary_structure(2, 2)  # 2D, 8-connectivity
    labeled_shoots, num_shoots = ndimage.label(shoot_mask, structure=structure)
    
    ref_points = {}
    
    for label in range(1, num_shoots + 1):
        shoot_region = labeled_shoots == label
        y_coords, x_coords = np.where(shoot_region)
        
        if len(y_coords) == 0:
            continue
            
        # Centroid
        centroid_y = np.mean(y_coords)
        centroid_x = np.mean(x_coords)
        
        # Bounding box
        min_y, max_y = y_coords.min(), y_coords.max()
        min_x, max_x = x_coords.min(), x_coords.max()
        
        # Bottom center: centroid x, maximum y (lowest point)
        bottom_center = (max_y, centroid_x)
        
        # Bottom most: actual bottom-most pixel
        bottom_idx = np.argmax(y_coords)
        bottom_most = (y_coords[bottom_idx], x_coords[bottom_idx])
        
        ref_points[label] = {
            'centroid': (centroid_y, centroid_x),
            'bottom_center': bottom_center,
            'bottom_most': bottom_most,
            'bbox': (min_y, max_y, min_x, max_x)
        }
    
    return labeled_shoots, num_shoots, ref_points


def filter_root_mask_by_sampling_box(root_mask, shoot_mask, ref_points, num_shoots,
                                     sampling_buffer_above=200,
                                     sampling_buffer_below=300):
    """Filter root mask to keep only components starting within sampling box.
    
    Creates a sampling box around all shoots and keeps only skeleton components
    whose top (min_y) falls within this box. Preserves full component length even
    if it extends beyond the box boundaries.
    
    Args:
        root_mask: Binary mask array with root regions (H, W)
        shoot_mask: Binary mask array with shoot regions (H, W)
        ref_points: Dict from get_shoot_reference_points_multi
        num_shoots: Number of shoot regions
        sampling_buffer_above: Pixels above top shoot to include
        sampling_buffer_below: Pixels below bottom shoot to include
        
    Returns:
        np.ndarray: Binary mask with filtered root components, preserving full lengths
        
    Examples:
        >>> filtered_mask = filter_root_mask_by_sampling_box(
        ...     root_mask, shoot_mask, ref_points, 5
        ... )
        >>> structures = extract_root_structures(filtered_mask)
    """
    # Calculate sampling box
    shoot_y_min = min(ref_points[s]['bbox'][0] for s in range(1, num_shoots + 1))
    shoot_y_max = max(ref_points[s]['bbox'][1] for s in range(1, num_shoots + 1))
    
    sampling_box_top = max(0, shoot_y_min - sampling_buffer_above)
    sampling_box_bottom = min(root_mask.shape[0], shoot_y_max + sampling_buffer_below)
    
    # Label all connected components in the FULL mask
    labeled_mask = label_components(root_mask)
    num_components = labeled_mask.max()
    
    print(f"Sampling box: rows {sampling_box_top:.0f} to {sampling_box_bottom:.0f}")
    print(f"Found {num_components} components in full mask")
    
    # Check which components have their TOP (min_y) in the sampling box
    valid_labels = set()
    
    for component_label in range(1, num_components + 1):
        component_mask = labeled_mask == component_label
        coords = np.argwhere(component_mask)
        
        if len(coords) == 0:
            continue
        
        comp_top_y = coords[:, 0].min()
        
        # Keep component if its top is in the sampling box
        if sampling_box_top <= comp_top_y <= sampling_box_bottom:
            valid_labels.add(component_label)
    
    # Create filtered mask with FULL components (not cropped)
    filtered_mask = np.isin(labeled_mask, list(valid_labels))
    
    print(f"Kept {len(valid_labels)} components (full length preserved)")
    
    return filtered_mask.astype(bool)


def score_skeleton_for_shoot(component_props, shoot_ref_point, shoot_bbox,
                              max_distance=100, max_size=1000, ideal_aspect=5.0,
                              distance_weight=0.5, size_weight=0.2, verticality_weight=0.3,
                              max_horizontal_offset=400,
                              max_above_shoot=200,
                              max_below_shoot=100,
                              min_skeleton_pixels=6,  # NEW parameter
                              edge_exclusion_zones=[(0, 900), (3300, 4000)]):
    """Score how likely a skeleton component is to be a root for a given shoot.
    
    Applies spatial filters (horizontal distance, edge zones) and calculates a
    combined score based on proximity, size, and verticality. Higher scores
    indicate better matches.
    
    Args:
        component_props: Dict with keys:
            - 'label': Component identifier
            - 'centroid': (y, x) tuple
            - 'bbox': (min_y, max_y, min_x, max_x) tuple
            - 'num_pixels': Total skeleton pixels
            - 'vertical_extent': Height in pixels
            - 'horizontal_extent': Width in pixels
        shoot_ref_point: (y, x) tuple for shoot reference position
        shoot_bbox: (min_y, max_y, min_x, max_x) tuple for shoot bounding box
        max_distance: Distance normalization constant (pixels)
        max_size: Size normalization constant (pixels)
        ideal_aspect: Ideal vertical:horizontal ratio for roots
        distance_weight: Weight for proximity score (0-1)
        size_weight: Weight for skeleton length (0-1)
        verticality_weight: Weight for vertical orientation (0-1)
        max_horizontal_offset: Maximum horizontal distance from shoot (pixels)
        max_above_shoot: Allow roots to extend this many pixels above shoot bottom
        max_below_shoot: Maximum distance below shoot bottom where root can start (pixels)
        min_skeleton_pixels: Minimum skeleton pixels to be valid candidate  # ADDED
        edge_exclusion_zones: List of (min_x, max_x) tuples for edge noise
        
    Returns:
        dict: Scoring results with keys:
            - 'score': Combined score (higher is better), -1 if invalid
            - 'distance': Euclidean distance from shoot
            - 'distance_score': Normalized distance component
            - 'size_score': Normalized size component
            - 'verticality_score': Normalized verticality component
            - 'aspect_ratio': Vertical:horizontal ratio
            - 'reason': Rejection reason if score is -1
            
    Examples:
        >>> score_result = score_skeleton_for_shoot(comp_props, shoot_ref, shoot_bbox)
        >>> if score_result['score'] > 0:
        ...     print(f"Valid candidate with score {score_result['score']:.3f}")
    """
    shoot_y, shoot_x = shoot_ref_point
    shoot_bottom = shoot_bbox[1]
    
    comp_centroid_y, comp_centroid_x = component_props['centroid']
    comp_top_y = component_props['bbox'][0]
    comp_bottom_y = component_props['bbox'][1]
    comp_bbox = component_props['bbox']
    comp_pixels = component_props['num_pixels']
    vertical_extent = component_props['vertical_extent']
    horizontal_extent = component_props['horizontal_extent']

    # FILTER 0: Minimum size filter (reject noise)
    if comp_pixels < min_skeleton_pixels:
        return {
            'score': -1,
            'distance': float('inf'),
            'distance_score': 0,
            'size_score': 0,
            'verticality_score': 0,
            'reason': 'too_small'
        }

    # FILTER 1: Root must extend reasonably below shoot
    if comp_bottom_y < (shoot_bottom - max_above_shoot):
        return {
            'score': -1,
            'distance': float('inf'),
            'distance_score': 0,
            'size_score': 0,
            'verticality_score': 0,
            'reason': 'above_shoot'
        }
    
    # FILTER 2: Root must start within reasonable distance below shoot
    if comp_top_y > (shoot_bottom + max_below_shoot):
        return {
            'score': -1,
            'distance': float('inf'),
            'distance_score': 0,
            'size_score': 0,
            'verticality_score': 0,
            'reason': 'too_far_below_shoot'
        }
    
    # FILTER 3: Horizontal overlap check
    comp_min_x = comp_bbox[2]
    comp_max_x = comp_bbox[3]
    search_zone_min_x = shoot_x - max_horizontal_offset
    search_zone_max_x = shoot_x + max_horizontal_offset
    
    if comp_max_x < search_zone_min_x or comp_min_x > search_zone_max_x:
        return {
            'score': -1,
            'distance': float('inf'),
            'distance_score': 0,
            'size_score': 0,
            'verticality_score': 0,
            'reason': f'too_far_horizontally'
        }
    
    # FILTER 4: Edge exclusion zones
    for min_edge_x, max_edge_x in edge_exclusion_zones:
        if min_edge_x <= comp_centroid_x <= max_edge_x:
            return {
                'score': -1,
                'distance': float('inf'),
                'distance_score': 0,
                'size_score': 0,
                'verticality_score': 0,
                'reason': f'in_edge_zone'
            }
    
    # Calculate scores
    distance = np.sqrt((comp_top_y - shoot_y)**2 + (comp_centroid_x - shoot_x)**2)
    distance_score = 1.0 / (1.0 + distance / max_distance)
    
    size_score = min(comp_pixels / max_size, 1.0)
    
    if horizontal_extent > 0:
        aspect_ratio = vertical_extent / horizontal_extent
        verticality_score = min(aspect_ratio / ideal_aspect, 1.0)
    else:
        verticality_score = 1.0
    
    combined = (distance_weight * distance_score + 
                size_weight * size_score + 
                verticality_weight * verticality_score)
    
    return {
        'score': combined,
        'distance': distance,
        'distance_score': distance_score,
        'size_score': size_score,
        'verticality_score': verticality_score,
        'aspect_ratio': vertical_extent / max(horizontal_extent, 1)
    }


def find_valid_candidates_for_shoots(structures, ref_points, num_shoots, config=None):
    """Find and score valid root candidates for each shoot.
    
    Uses sampling box pre-filtering for efficiency, then scores each component
    for each shoot. Returns organized candidates sorted by score.
    
    Args:
        structures: Dict from extract_root_structures with keys:
            - 'skeleton': Full skeleton array
            - 'labeled_skeleton': Labeled skeleton array
            - 'unique_labels': Array of label IDs
            - 'roots': Dict of root data by label
        ref_points: Dict from get_shoot_reference_points_multi
        num_shoots: Number of shoots (typically 5)
        config: RootMatchingConfig instance (uses defaults if None)
        
    Returns:
        dict: {shoot_label: [(root_label, score_dict), ...]} where each shoot's
              list is sorted by score descending
              
    Examples:
        >>> candidates = find_valid_candidates_for_shoots(structures, ref_points, 5)
        >>> for shoot_label in range(1, 6):
        ...     print(f"Shoot {shoot_label}: {len(candidates[shoot_label])} candidates")
    """
    if config is None:
        config = RootMatchingConfig()
    
    # Storage for each shoot's candidates
    shoot_candidates = {label: [] for label in range(1, num_shoots + 1)}
    
    total_components = len(structures['roots'])
    
    # Process each skeleton component
    for root_label, root_data in structures['roots'].items():
        branch_data = root_data.get('branch_data')
        
        # Skip if no branch data
        if branch_data is None or len(branch_data) == 0:
            continue
        
        # Extract coordinates for this component
        root_mask_region = root_data['mask']
        coords = np.argwhere(root_mask_region)
        
        if len(coords) == 0:
            continue
        
        # Build component properties
        comp_props = {
            'label': root_label,
            'centroid': (coords[:, 0].mean(), coords[:, 1].mean()),
            'bbox': (coords[:, 0].min(), coords[:, 0].max(), 
                    coords[:, 1].min(), coords[:, 1].max()),
            'num_pixels': root_data['total_pixels'],
            'vertical_extent': coords[:, 0].max() - coords[:, 0].min(),
            'horizontal_extent': coords[:, 1].max() - coords[:, 1].min()
        }
        
        # Score this component for each shoot
        for shoot_label in range(1, num_shoots + 1):
            shoot_ref = ref_points[shoot_label]['bottom_center']
            shoot_bbox = ref_points[shoot_label]['bbox']
            
            score_result = score_skeleton_for_shoot(
                comp_props, shoot_ref, shoot_bbox,
                max_distance=config.max_distance,
                max_size=config.max_size,
                ideal_aspect=config.ideal_aspect,
                distance_weight=config.distance_weight,
                size_weight=config.size_weight,
                verticality_weight=config.verticality_weight,
                max_horizontal_offset=config.max_horizontal_offset,
                max_above_shoot=config.max_above_shoot,
                max_below_shoot=config.max_below_shoot,
                min_skeleton_pixels=config.min_skeleton_pixels,  # ADDED
                edge_exclusion_zones=config.edge_exclusion_zones
            )
            
            if score_result['score'] > 0:
                shoot_candidates[shoot_label].append((root_label, score_result))
    
    # Sort each shoot's candidates by score (descending)
    for shoot_label in shoot_candidates:
        shoot_candidates[shoot_label].sort(key=lambda x: x[1]['score'], reverse=True)
    
    print(f"Valid candidates per shoot:")
    for shoot_label in range(1, num_shoots + 1):
        print(f"  Shoot {shoot_label}: {len(shoot_candidates[shoot_label])} candidates")
    
    return shoot_candidates



def assign_roots_to_shoots_greedy(shoot_candidates, ref_points, num_shoots, config=None, min_score_threshold=0.3):
    """Assign roots to shoots using greedy left-to-right algorithm.
    
    Processes shoots in left-to-right order, assigning the best available
    (unassigned) root to each shoot. Guarantees exactly num_shoots outputs.
    
    Args:
        shoot_candidates: Dict from find_valid_candidates_for_shoots with structure
            {shoot_label: [(root_label, score_dict), ...]}
        ref_points: Dict from get_shoot_reference_points_multi
        num_shoots: Number of shoots (always 5)
        config: RootMatchingConfig instance (optional)
        min_score_threshold: Minimum score required for assignment (default 0.3)
        
    Returns:
        dict: {shoot_label: {
            'root_label': int or None,
            'score_dict': dict or None,
            'order': int  # 0=leftmost, 4=rightmost
        }}
    """
    # Sort shoots by x-position (left to right)
    shoot_positions = []
    for shoot_label in range(1, num_shoots + 1):
        centroid_x = ref_points[shoot_label]['centroid'][1]
        shoot_positions.append((shoot_label, centroid_x))
    
    shoot_positions.sort(key=lambda x: x[1])  # Sort by x coordinate
    
    # Track assigned roots
    assigned_roots = set()
    
    # Build assignments
    assignments = {}
    
    for order, (shoot_label, _) in enumerate(shoot_positions):
        candidates = shoot_candidates[shoot_label]
        
        # Find best unassigned root
        best_root = None
        best_score_dict = None
        
        for root_label, score_dict in candidates:
            if root_label not in assigned_roots:
                # Check if score meets minimum threshold
                if score_dict['score'] >= min_score_threshold:
                    best_root = root_label
                    best_score_dict = score_dict
                    break  # Candidates are already sorted by score
        
        # Store assignment
        if best_root is not None:
            assignments[shoot_label] = {
                'root_label': best_root,
                'score_dict': best_score_dict,
                'order': order
            }
            assigned_roots.add(best_root)
        else:
            # No valid candidates (ungerminated seed or below threshold)
            assignments[shoot_label] = {
                'root_label': None,
                'score_dict': None,
                'order': order
            }
    
    return assignments


def find_best_top_node(root_skeleton, shoot_ref_point, structures, root_label, 
                       distance_threshold=150, prefer_topmost=True):
    """Find the skeleton node closest to the shoot reference point.
    
    Uses a hybrid approach: filters nodes within distance_threshold of shoot,
    then selects based on preference (topmost or closest).
    
    Args:
        root_skeleton: Binary mask for this specific root (not used, kept for signature)
        shoot_ref_point: (y, x) tuple for shoot reference position
        structures: Dict from extract_root_structures
        root_label: Label ID for this root
        distance_threshold: Maximum distance from shoot to consider nodes (pixels)
        prefer_topmost: If True, pick topmost among candidates; if False, pick closest
        
    Returns:
        int: Node ID of the selected top node, or None if no valid nodes found
    """
    
    # Get branch data for this root
    branch_data = structures['roots'][root_label].get('branch_data')
    if branch_data is None or len(branch_data) == 0:
        return None
    
    # Extract all unique nodes from branch_data
    src_nodes = branch_data[['node-id-src', 'image-coord-src-0', 'image-coord-src-1']].drop_duplicates()
    dst_nodes = branch_data[['node-id-dst', 'image-coord-dst-0', 'image-coord-dst-1']].drop_duplicates()
    
    src_nodes.columns = ['node_id', 'y', 'x']
    dst_nodes.columns = ['node_id', 'y', 'x']
    
    all_nodes = pd.concat([src_nodes, dst_nodes]).drop_duplicates(subset='node_id')
    node_coords = all_nodes[['y', 'x']].values
    node_ids = all_nodes['node_id'].values
    
    if len(node_ids) == 0:
        return None
    
    # Calculate distances to shoot reference point
    distances = cdist(node_coords, [shoot_ref_point], metric='euclidean').flatten()
    
    # Filter nodes within distance threshold
    within_threshold = distances <= distance_threshold
    
    if not np.any(within_threshold):
        # No nodes within threshold, fall back to closest node
        return int(node_ids[np.argmin(distances)])
    
    # Get candidate nodes within threshold
    candidate_indices = np.where(within_threshold)[0]
    candidate_coords = node_coords[candidate_indices]
    candidate_ids = node_ids[candidate_indices]
    candidate_distances = distances[candidate_indices]
    
    if prefer_topmost:
        # Pick node with lowest y-coordinate (topmost)
        best_idx = np.argmin(candidate_coords[:, 0])
        return int(candidate_ids[best_idx])
    else:
        # Pick node with minimum distance
        best_idx = np.argmin(candidate_distances)
        return int(candidate_ids[best_idx])
    

def pixel_to_robot_coords(pixel_x, pixel_y, image_shape, 
                         dish_size_m=0.15,
                         dish_offset_m=[0.10775, 0.062, 0.175]):
    """
    Convert pixel coordinates to robot coordinates in meters.
    
    Converts ROI-relative pixel coordinates to robot world coordinates.
    Assumes square petri dish with gantry axes aligned to image axes:
    - Robot X-axis aligns with image Y-axis (rows, downward)
    - Robot Y-axis aligns with image X-axis (columns, rightward)
    
    Args:
        pixel_x: X coordinate in pixels (column, increases right)
        pixel_y: Y coordinate in pixels (row, increases down)
        image_shape: Tuple of (height, width) of the ROI
        dish_size_m: Size of square petri dish in meters (default 0.15m)
        dish_offset_m: Robot coordinates [x, y, z] of plate top-left corner in meters
                      Default [0.10775, 0.062, 0.175] from simulation specification
        
    Returns:
        tuple: (robot_x_m, robot_y_m, robot_z_m) in meters
    """
    height, width = image_shape[:2]
    
    # Calculate scale (meters per pixel)
    scale = dish_size_m / width
    
    # Convert to plate-relative meters
    plate_x = pixel_x * scale
    plate_y = pixel_y * scale
    
    # Map to robot coordinates (gantry alignment, no rotation)
    robot_x = plate_y + dish_offset_m[0]  # Image Y-axis → Robot X-axis
    robot_y = plate_x + dish_offset_m[1]  # Image X-axis → Robot Y-axis
    robot_z = dish_offset_m[2]
    
    return robot_x, robot_y, robot_z

def match_roots_to_shoots_complete(shoot_mask, root_mask, image_path, config=None, 
                                   distance_threshold=150, prefer_topmost=True,
                                   min_score_threshold=0.3, verbose=False,
                                   return_dataframe=False, sample_idx=None,
                                   visual_debugging=False, output_path=None):
    """Complete pipeline: shoot and root masks in, 5 length measurements out.
    
    High-level wrapper that runs the entire matching pipeline:
    1. Extract shoot reference points
    2. Filter root mask by sampling box
    3. Extract root structures
    4. Find valid candidates for each shoot
    5. Assign roots to shoots greedily (left to right)
    6. Find best top nodes for assignments
    7. Calculate root lengths from top nodes
    
    Args:
        shoot_mask: Binary mask array with shoot regions (H, W)
        root_mask: Binary mask array with root regions (H, W)
        image_path: Path to original image for ROI detection
        config: RootMatchingConfig instance (uses defaults if None)
        distance_threshold: Max distance from shoot for top node selection (pixels)
        prefer_topmost: If True, prefer topmost node; if False, prefer closest
        min_score_threshold: Minimum score for valid assignment
        verbose: Print progress messages
        return_dataframe: If True, return pandas DataFrame; if False, return numpy array
        sample_idx: Sample index to include in DataFrame (only used if return_dataframe=True)
        visual_debugging: If True, display visualizations of assignments and root lengths
        output_path: Optional path to save visualization outputs (not yet implemented)
        
    Returns:
        If return_dataframe=False (default):
            np.ndarray: Array of 5 root lengths in pixels, ordered left to right.
        If return_dataframe=True:
            pd.DataFrame: DataFrame with columns:
                - 'plant_order': 1-5 (left to right)
                - 'Plant ID': test_image_{sample_idx + 1}
                - 'Length (px)': root length in pixels
                - 'length_px': root length in pixels (duplicate)
                - 'top_node_x': x-coordinate of top node (pixels)
                - 'top_node_y': y-coordinate of top node (pixels)
                - 'endpoint_x': x-coordinate of endpoint node (pixels)
                - 'endpoint_y': y-coordinate of endpoint node (pixels)
                - 'top_node_robot_x': x-coordinate of top node (meters)
                - 'top_node_robot_y': y-coordinate of top node (meters)
                - 'top_node_robot_z': z-coordinate of top node (meters)
                - 'endpoint_robot_x': x-coordinate of endpoint (meters)
                - 'endpoint_robot_y': y-coordinate of endpoint (meters)
                - 'endpoint_robot_z': z-coordinate of endpoint (meters)
            
    Raises:
        ValueError: If shoot_mask does not contain exactly 5 shoots (warning only)
        
    Examples:
        >>> # Get numpy array
        >>> lengths = match_roots_to_shoots_complete(shoot_mask, root_mask)
        >>> 
        >>> # Get DataFrame
        >>> df = match_roots_to_shoots_complete(shoot_mask, root_mask, 
        ...                                      return_dataframe=True, sample_idx=0)
        
    Notes:
        - Always returns exactly 5 measurements
        - Left-to-right order is determined by shoot x-position (centroid)
        - Robust to missing roots (returns 0.0) and noisy masks
    """

    
    if config is None:
        config = RootMatchingConfig()
    
    if verbose:
        print("Starting complete root-shoot matching pipeline...")
    
    # Step 0: Detect ROI from original image
    if verbose:
        print("  Step 0: Detecting ROI from original image...")
    
    from library.roi import detect_roi
    from library.mask_processing import load_mask
    
    roi_bbox = detect_roi(load_mask(str(image_path)))
    (x1, y1), (x2, y2) = roi_bbox
    roi_width = x2 - x1
    roi_height = y2 - y1
    
    if verbose:
        print(f"    ROI bbox: ({x1}, {y1}) to ({x2}, {y2}), size={roi_width}x{roi_height}px")
    
    # Step 1: Get shoot reference points
    if verbose:
        print("  Step 1: Extracting shoot reference points...")
    labeled_shoots, num_shoots, ref_points = get_shoot_reference_points_multi(shoot_mask)
    
    if num_shoots != 5:
        warnings.warn(f"Expected 5 shoots, found {num_shoots}. Continuing anyway.", UserWarning)
    
    if verbose:
        print(f"    Found {num_shoots} shoots")
    
    # Step 2: Filter root mask by sampling box
    if verbose:
        print("  Step 2: Filtering root mask by sampling box...")
    filtered_root_mask = filter_root_mask_by_sampling_box(
        root_mask, shoot_mask, ref_points, num_shoots,
        sampling_buffer_above=config.sampling_buffer_above,
        sampling_buffer_below=config.sampling_buffer_below
    )
    
    # Step 3: Extract root structures
    if verbose:
        print("  Step 3: Extracting root structures...")
    structures = extract_root_structures(filtered_root_mask, verbose=verbose)
    
    if verbose:
        print(f"    Extracted {len(structures['roots'])} root structures")
    
    # Step 4: Find valid candidates
    if verbose:
        print("  Step 4: Finding valid candidates for each shoot...")
    shoot_candidates = find_valid_candidates_for_shoots(structures, ref_points, num_shoots, config)
    
    # Step 5: Assign roots to shoots
    if verbose:
        print("  Step 5: Assigning roots to shoots (greedy left-to-right)...")
    assignments = assign_roots_to_shoots_greedy(shoot_candidates, ref_points, num_shoots, 
                                                min_score_threshold=min_score_threshold)
    
    # Step 6 & 7: Find top nodes and calculate lengths
    if verbose:
        print("  Step 6-7: Finding top nodes and calculating lengths...")
    
    # Helper function to get node coordinates
    def get_node_coords(node_id, branch_data):
        """Extract (y, x) coordinates for a given node_id from branch_data."""
        for idx, row in branch_data.iterrows():
            if row['node-id-src'] == node_id:
                return (row['image-coord-src-0'], row['image-coord-src-1'])
            elif row['node-id-dst'] == node_id:
                return (row['image-coord-dst-0'], row['image-coord-dst-1'])
        return None
    
    # Build structure for process_matched_roots_to_lengths
    top_node_results = {}
    
    for shoot_label, info in assignments.items():
        root_label = info['root_label']
        
        if root_label is not None:
            # Find top node
            shoot_ref = ref_points[shoot_label]['bottom_center']
            top_node = find_best_top_node(None, shoot_ref, structures, root_label,
                                         distance_threshold=distance_threshold,
                                         prefer_topmost=prefer_topmost)
            
            if top_node is not None:
                branch_data = structures['roots'][root_label]['branch_data']
                
                top_node_results[root_label] = {
                    'shoot_label': shoot_label,
                    'top_nodes': [(top_node, 0)],
                    'branch_data': branch_data
                }
    
    # Calculate lengths and coordinates for all matched roots
    lengths_dict = {}
    top_coords_dict = {}
    endpoint_coords_dict = {}
    
    for root_label, result in top_node_results.items():
        branch_data = result['branch_data']
        shoot_label = result['shoot_label']
        top_node = result['top_nodes'][0][0]
        
        # Get top node coordinates
        top_coords = get_node_coords(top_node, branch_data)
        if top_coords:
            top_coords_dict[shoot_label] = top_coords
        
        try:
            # Find longest path from top node
            path = find_farthest_endpoint_path(
                branch_data,
                top_node,
                direction='down',
                use_smart_scoring=True,
                verbose=False
            )
            
            # Get endpoint node (last node in path)
            endpoint_node = path[-1][0]
            endpoint_coords = get_node_coords(endpoint_node, branch_data)
            if endpoint_coords:
                endpoint_coords_dict[shoot_label] = endpoint_coords
            
            # Calculate length
            root_length = calculate_skeleton_length_px(path)
            lengths_dict[shoot_label] = root_length
            
            if verbose:
                print(f"    Shoot {shoot_label}: Root {root_label}, length={root_length:.1f}px, "
                      f"top={top_coords}, endpoint={endpoint_coords}")
            
        except Exception as e:
            warnings.warn(f"Failed to calculate length for root {root_label} (shoot {shoot_label}): {e}", UserWarning)
            lengths_dict[shoot_label] = 0.0
            
            if verbose:
                print(f"    Shoot {shoot_label}: Root {root_label}, ERROR - returning 0.0")
    
    # Build output array ordered by shoot position (left to right)
    shoot_positions = []
    for shoot_label in range(1, num_shoots + 1):
        shoot_x = ref_points[shoot_label]['centroid'][1]
        shoot_positions.append((shoot_label, shoot_x))
    
    shoot_positions.sort(key=lambda x: x[1])
    
    # Create final arrays
    lengths_array = np.array([
        lengths_dict.get(shoot_label, 0.0)
        for shoot_label, _ in shoot_positions
    ])
    
    top_x_array = np.array([
        top_coords_dict.get(shoot_label, (np.nan, np.nan))[1]
        for shoot_label, _ in shoot_positions
    ])
    
    top_y_array = np.array([
        top_coords_dict.get(shoot_label, (np.nan, np.nan))[0]
        for shoot_label, _ in shoot_positions
    ])
    
    endpoint_x_array = np.array([
        endpoint_coords_dict.get(shoot_label, (np.nan, np.nan))[1]
        for shoot_label, _ in shoot_positions
    ])
    
    endpoint_y_array = np.array([
        endpoint_coords_dict.get(shoot_label, (np.nan, np.nan))[0]
        for shoot_label, _ in shoot_positions
    ])
    
    if verbose:
        print(f"\n  Final lengths (left to right): {lengths_array}")
        print("  Pipeline complete!")
    
    # Visual debugging
    if visual_debugging:

        im_path = Path(image_path)
        print("="*50)
        print(f"\n  Generating visualizations for {im_path.name}")

        print('   Original image')
        image = load_mask(im_path)

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image, cmap='gray')
        ax.set_title(im_path.name, fontsize=14)
        ax.axis('off')
        plt.tight_layout()
        plt.show()

        
        from library.root_analysis_visualization import visualize_assignments, visualize_root_lengths
        
        # Visualize assignments
        print('    Visualize shoot & soot assignments')
        visualize_assignments(shoot_mask, root_mask, structures, assignments, ref_points)
        
        # Visualize root lengths with detailed views
        print('    Visualize root skeletons with node and edge network')
        visualize_root_lengths(structures, top_node_results, labeled_shoots, 
                              show_detailed_roots=True)
    
    # Return DataFrame or array
    if return_dataframe:
        import pandas as pd
        
        # Convert pixel coordinates to robot coordinates using ROI-relative system
        roi_shape = (roi_height, roi_width)
        
        top_robot_coords = [pixel_to_robot_coords(x - x1, y - y1, roi_shape) 
                           if not (np.isnan(x) or np.isnan(y)) else (np.nan, np.nan, np.nan)
                           for x, y in zip(top_x_array, top_y_array)]
        endpoint_robot_coords = [pixel_to_robot_coords(x - x1, y - y1, roi_shape) 
                                if not (np.isnan(x) or np.isnan(y)) else (np.nan, np.nan, np.nan)
                                for x, y in zip(endpoint_x_array, endpoint_y_array)]
        
        # Unpack into separate arrays
        top_robot_x = np.array([c[0] for c in top_robot_coords])
        top_robot_y = np.array([c[1] for c in top_robot_coords])
        top_robot_z = np.array([c[2] for c in top_robot_coords])
        
        endpoint_robot_x = np.array([c[0] for c in endpoint_robot_coords])
        endpoint_robot_y = np.array([c[1] for c in endpoint_robot_coords])
        endpoint_robot_z = np.array([c[2] for c in endpoint_robot_coords])
        
        df_data = {
            'plant_order': list(range(1, 6)),
            'Plant ID': [f'test_image_{sample_idx + 1:02d}_plant_{i}' if sample_idx is not None else f'unknown_plant_{i}' for i in range(1, 6)],
            'Length (px)': lengths_array,
            'length_px': lengths_array,
            'top_node_x': top_x_array,
            'top_node_y': top_y_array,
            'endpoint_x': endpoint_x_array,
            'endpoint_y': endpoint_y_array,
            'top_node_robot_x': top_robot_x,
            'top_node_robot_y': top_robot_y,
            'top_node_robot_z': top_robot_z,
            'endpoint_robot_x': endpoint_robot_x,
            'endpoint_robot_y': endpoint_robot_y,
            'endpoint_robot_z': endpoint_robot_z
        }
        
        return pd.DataFrame(df_data)
    else:
        return lengths_array