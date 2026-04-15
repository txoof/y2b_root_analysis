from scipy.ndimage import binary_dilation
from scipy import ndimage

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib.patches as mpatches
from library.root_analysis import *


def visualize_skeleton_with_nodes(skeleton, show_branch_labels=False, row_range=None, padding=50, dilate_iterations=3):
    """
    Visualize skeleton with nodes and optionally branch labels.

    Args:
        skeleton (skeleton numpy array): object to show
        show_branch_labels (bool): add labels
        row_range(tuple): rows to zoom in on
        padding(int): add additional pixels to make skeleton easier to view
        dilate_iterations(int): number of rounds to dilate and increase size

    Returns:
        branch_data (pandas table)
    """
    
    # Thicken skeleton for visibility
    if dilate_iterations > 0:
        thick_skeleton = binary_dilation(skeleton, iterations=dilate_iterations)
        skeleton_uint8 = (thick_skeleton * 255).astype(np.uint8)
    else:
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    
    # Find bounding box
    rows, cols = np.where(skeleton)
    
    if row_range is not None:
        row_min, row_max = row_range
        mask = (rows >= row_min) & (rows <= row_max)
        if np.any(mask):
            col_min, col_max = cols[mask].min(), cols[mask].max()
        else:
            col_min, col_max = cols.min(), cols.max()
    else:
        row_min, row_max = rows.min(), rows.max()
        col_min, col_max = cols.min(), cols.max()
    
    # Get branch data and nodes
    skeleton_obj = Skeleton(skeleton)
    branch_data = summarize(skeleton_obj)
    
    # Get node coordinates
    node_coords = {}
    for idx, row in branch_data.iterrows():
        node_coords[row['node-id-src']] = (row['coord-src-0'], row['coord-src-1'])
        node_coords[row['node-id-dst']] = (row['coord-dst-0'], row['coord-dst-1'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.imshow(skeleton_uint8, cmap='gray')
    
    # Track occupied regions (for collision detection)
    occupied_regions = []  # List of (row, col, radius) tuples
    
    # First pass: plot node markers
    visible_nodes = []
    for node_id, (node_row, node_col) in node_coords.items():
        if (row_min - padding <= node_row <= row_max + padding and 
            col_min - padding <= node_col <= col_max + padding):
            visible_nodes.append((node_id, node_row, node_col))
            ax.plot(node_col, node_row, 'ro', markersize=12, zorder=3)
            # Mark node position as occupied
            occupied_regions.append((node_row, node_col, 15))  # Small radius for node dot itself
    
    # Second pass: place node labels with collision avoidance
    for node_id, node_row, node_col in visible_nodes:
        # Try different offset positions for node label
        offsets = [(40, 0), (-40, 0), (0, 40), (0, -40), (30, 30), (-30, -30), (30, -30), (-30, 30)]
        best_offset = (40, 0)
        min_collision = float('inf')
        
        for offset_col, offset_row in offsets:
            test_col = node_col + offset_col
            test_row = node_row + offset_row
            
            # Check collision with all occupied regions
            max_collision = 0
            for occ_row, occ_col, radius in occupied_regions:
                dist = np.sqrt((test_row - occ_row)**2 + (test_col - occ_col)**2)
                if dist < radius:
                    max_collision = max(max_collision, radius - dist)
            
            if max_collision < min_collision:
                min_collision = max_collision
                best_offset = (offset_col, offset_row)
                if max_collision == 0:
                    break
        
        label_col = node_col + best_offset[0]
        label_row = node_row + best_offset[1]
        
        # Draw leader line
        ax.plot([node_col, label_col], [node_row, label_row], 
               'y--', linewidth=1, alpha=0.6, zorder=2)
        
        # Draw label
        ax.text(label_col, label_row, str(int(node_id)), color='yellow', fontsize=11, 
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               ha='center', zorder=4)
        
        # Mark label area as occupied
        occupied_regions.append((label_row, label_col, 45))
    
    # Branch labels with collision avoidance
    if show_branch_labels:
        branch_positions = []
        for idx, row in branch_data.iterrows():
            mid_row = (row['coord-src-0'] + row['coord-dst-0']) / 2
            mid_col = (row['coord-src-1'] + row['coord-dst-1']) / 2
            
            if (row_min - padding <= mid_row <= row_max + padding and 
                col_min - padding <= mid_col <= col_max + padding):
                branch_positions.append((idx, mid_row, mid_col))
        
        for idx, mid_row, mid_col in branch_positions:
            # Try different offset positions
            offsets = [(60, 0), (-60, 0), (0, 60), (0, -60), (45, 45), (-45, -45), 
                      (120, 0), (-120, 0), (0, 120), (0, -120)]
            best_offset = (60, 0)
            min_collision = float('inf')
            
            for offset_col, offset_row in offsets:
                test_col = mid_col + offset_col
                test_row = mid_row + offset_row
                
                max_collision = 0
                for occ_row, occ_col, radius in occupied_regions:
                    dist = np.sqrt((test_row - occ_row)**2 + (test_col - occ_col)**2)
                    if dist < radius:
                        max_collision = max(max_collision, radius - dist)
                
                if max_collision < min_collision:
                    min_collision = max_collision
                    best_offset = (offset_col, offset_row)
                    if max_collision == 0:
                        break
            
            label_col = mid_col + best_offset[0]
            label_row = mid_row + best_offset[1]
            
            # Draw leader line
            ax.plot([mid_col, label_col], [mid_row, label_row], 
                   'c--', linewidth=1, alpha=0.5, zorder=1)
            
            # Draw label
            ax.text(label_col, label_row, f"B{idx}", color='cyan', fontsize=9, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7),
                   ha='center', zorder=2)
            
            occupied_regions.append((label_row, label_col, 40))
    
    # Set zoom limits
    ax.set_xlim(col_min - padding, col_max + padding)
    ax.set_ylim(row_max + padding, row_min - padding)
    
    title = f'Skeleton with {len(node_coords)} nodes, {branch_data.shape[0]} branches'
    if row_range:
        title += f' (rows {row_range[0]}-{row_range[1]})'
    ax.set_title(title)
    
    plt.show()
    
    return branch_data

def visualize_trunk_path(skeleton, branch_data, trunk_path, title='Trunk Path', 
                        dilate_iterations=3, zoom_to_path=True, padding=100):
    """
    Visualize the trunk path overlaid on the skeleton image.
    
    Args:
        skeleton: Binary skeleton array
        branch_data: DataFrame with branch information
        trunk_path: List of tuples from trace_vertical_path [(node, branch_idx, slope, vertical), ...]
        title: Plot title
        dilate_iterations: Number of iterations to thicken skeleton
        zoom_to_path: Whether to zoom to the trunk path region
        padding: Padding around zoom region
    """
    
    # Thicken skeleton for visibility
    thick_skeleton = binary_dilation(skeleton, iterations=dilate_iterations)
    skeleton_uint8 = (thick_skeleton * 255).astype(np.uint8)
    
    # Get all nodes in the path
    path_nodes = [int(node) for node, _, _, _ in trunk_path]
    path_branches = [idx for _, idx, _, _ in trunk_path if idx is not None]
    
    # Get coordinates for all nodes
    all_node_coords = {}
    for idx, row in branch_data.iterrows():
        all_node_coords[row['node-id-src']] = (row['coord-src-0'], row['coord-src-1'])
        all_node_coords[row['node-id-dst']] = (row['coord-dst-0'], row['coord-dst-1'])
    
    # Get bounding box of path nodes for zooming
    if zoom_to_path and path_nodes:
        path_rows = [all_node_coords[n][0] for n in path_nodes if n in all_node_coords]
        path_cols = [all_node_coords[n][1] for n in path_nodes if n in all_node_coords]
        if path_rows and path_cols:
            row_min, row_max = min(path_rows), max(path_rows)
            col_min, col_max = min(path_cols), max(path_cols)
        else:
            zoom_to_path = False
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.imshow(skeleton_uint8, cmap='gray')
    
    # Draw all nodes (small, gray, semi-transparent)
    for node_id, (row, col) in all_node_coords.items():
        ax.plot(col, row, 'o', color='gray', markersize=4, alpha=0.3)
    
    # Draw trunk path connections FIRST (behind nodes)
    for i in range(len(path_nodes)-1):
        node1, node2 = path_nodes[i], path_nodes[i+1]
        if node1 in all_node_coords and node2 in all_node_coords:
            r1, c1 = all_node_coords[node1]
            r2, c2 = all_node_coords[node2]
            ax.plot([c1, c2], [r1, r2], 'r-', linewidth=4, alpha=0.8, zorder=2)
    
    # Draw trunk path nodes (large, colored)
    for i, node_id in enumerate(path_nodes):
        if node_id in all_node_coords:
            row, col = all_node_coords[node_id]
            # Color gradient: dark red (start) to bright yellow (end)
            color = plt.cm.autumn(i / max(len(path_nodes)-1, 1))
            ax.plot(col, row, 'o', color=color, markersize=16, zorder=3, 
                   markeredgecolor='white', markeredgewidth=2)
            ax.text(col+15, row, str(node_id), color='cyan', fontsize=12, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                   zorder=4)
    
    # Zoom to path if requested
    if zoom_to_path:
        ax.set_xlim(col_min - padding, col_max + padding)
        ax.set_ylim(row_max + padding, row_min - padding)
    
    # Title with path info
    path_str = " → ".join(map(str, path_nodes[:10]))  # First 10 nodes
    if len(path_nodes) > 10:
        path_str += "..."
    ax.set_title(f'{title}\nPath: {path_str}\n{title}', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Trunk path: {len(path_nodes)} nodes, {len(path_branches)} branches")
    print(f"Start node: {path_nodes[0]}, End node: {path_nodes[-1]}")
    
    # Calculate total length
    total_length = sum(branch_data.iloc[idx]['branch-distance'] 
                      for idx in path_branches)
    print(f"Total trunk length: {total_length:.2f} pixels")
    
    return total_length

def visualize_thickened_skeleton(binary_mask, dilate_iterations=3, figsize=(12, 8), show_labels=True):
    """
    Skeletonize, label, thicken, and visualize a binary mask.
    
    Args:
        binary_mask: Binary numpy array (bool or 0/255) or path to image file
        dilate_iterations: Number of dilation iterations to thicken skeleton
        figsize: Figure size tuple
        show_labels: Whether to show component labels on the image
    
    Returns:
        tuple: (skeleton, labeled_skeleton, unique_labels)
    """
    # Load from file if path provided
    if not isinstance(binary_mask, np.ndarray):
        binary_mask = cv2.imread(str(binary_mask), cv2.IMREAD_GRAYSCALE)

    
    # Get labeled skeleton
    skeleton, labeled_skeleton, unique_labels = label_skeleton(binary_mask)
    
    # Thicken skeleton for visibility
    thick_skeleton = binary_dilation(skeleton, iterations=dilate_iterations)
    thick_skeleton_uint8 = (thick_skeleton * 255).astype(np.uint8)
    
    # Visualize
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(thick_skeleton_uint8, cmap='gray')
    ax.set_title(f'{len(unique_labels)} skeleton(s) (thickened for visibility)')
    ax.axis('off')
    
    # Add labels for each structure
    if show_labels and len(unique_labels) > 0:
        for label_id in unique_labels:
            # Get all pixels for this label
            component_mask = (labeled_skeleton == label_id)
            rows, cols = np.where(component_mask)
            
            if len(rows) > 0:
                # Calculate centroid of this component
                centroid_row = int(np.mean(rows))
                centroid_col = int(np.mean(cols))
                
                # Find the bottommost point (highest row value)
                max_row = rows.max()
                
                # Place label below the bottommost point
                label_row = max_row + 50  # 50 pixels below
                label_col = centroid_col
                
                ax.text(label_col, label_row, str(int(label_id)), 
                       color='yellow', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='red', 
                                edgecolor='white', linewidth=2, alpha=0.8),
                       ha='center', va='center')
    
    plt.show()
    
    print(f"Found {len(unique_labels)} connected structure(s)")
    print(f"Total skeleton pixels: {np.sum(skeleton)}")
    
    return skeleton, labeled_skeleton, unique_labels


def visualize_assignments(shoot_mask, root_mask, structures, assignments, ref_points, size=(16, 8)):
    """Visualize shoot-root assignments with ordered labels 1-5 and root labels.
    
    Args:
        shoot_mask: Binary mask of shoot regions
        root_mask: Binary mask of root regions (not directly used)
        structures: Dictionary from extract_root_structures containing 'roots' with root data
        assignments: Dictionary from assign_roots_to_shoots_greedy with shoot assignments
        ref_points: Dictionary from get_shoot_reference_points_multi with shoot centroids
        size: Figure size tuple (width, height)
    
    Returns:
        None: Displays the visualization
    """
    # Label the shoot mask
    labeled_shoots, num_shoots = ndimage.label(shoot_mask)
    
    # Get image dimensions
    h, w = shoot_mask.shape
    
    # Create blank RGB image
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color palette (5 distinct colors)
    colors = [
        [255, 0, 0],      # Red
        [255, 165, 0],    # Orange  
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255]     # Magenta
    ]
    
    legend_elements = []
    
    # Process each shoot in order
    for shoot_label, info in assignments.items():
        order = info['order']
        root_label = info['root_label']
        color = colors[order]
        
        # Draw shoot
        shoot_region = (labeled_shoots == shoot_label)
        rgb_img[shoot_region] = color
        
        # Draw root if assigned
        if root_label is not None:
            root_mask = structures['roots'][root_label]['mask']
            dilated_root = ndimage.binary_dilation(root_mask, iterations=2)
            rgb_img[dilated_root] = color
            
            score = info['score_dict']['score']
            legend_elements.append(
                mpatches.Patch(color=np.array(color)/255, 
                             label=f"S{order+1}->R{root_label} ({score:.2f})")
            )
        else:
            legend_elements.append(
                mpatches.Patch(color=np.array(color)/255, 
                             label=f"S{order+1}->None")
            )
    
    # Create figure
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(rgb_img)
    
    # Add shoot order labels and root labels
    for shoot_label, info in assignments.items():
        order = info['order']
        root_label = info['root_label']
        
        if root_label is not None:
            # Find root endpoint (lowest y-coordinate)
            root_mask = structures['roots'][root_label]['mask']
            y_coords, x_coords = np.where(root_mask)
            
            if len(y_coords) > 0:
                max_y = np.max(y_coords)
                max_y_idx = np.argmax(y_coords)
                endpoint_x = x_coords[max_y_idx]
                
                # Shoot order label 50px below endpoint
                ax.text(endpoint_x, max_y + 300, str(order + 1), 
                       color='white', fontsize=20, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
        else:
            # Place label at shoot centroid for ungerminated
            cy, cx = ref_points[shoot_label]['centroid']
            ax.text(cx, cy, str(order + 1), 
                   color='white', fontsize=20, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(boxstyle='circle', facecolor='black', alpha=0.7))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    ax.set_title('Shoot-Root Assignments (Shoot# = Left->Right Order)', fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_root_shoot_matches(structures, labeled_shoots, root_matches, 
                                  zoom_to_content=True, padding=50, 
                                  root_thickness=3):
    """
    Visualize root-to-shoot matches with color coding.
    
    Args:
        structures: Dict from extract_root_structures
        labeled_shoots: Labeled shoot array
        root_matches: Dict from match_roots_to_shoots
        zoom_to_content: If True, zoom to bounding box of all content
        padding: Pixels to add around bounding box
        root_thickness: Pixels to dilate root skeletons for visibility
    """
    # Create RGB image
    h, w = structures['labeled_skeleton'].shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color map for shoots (distinct colors)
    colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
    ]
    
    # Draw shoots with their colors
    for shoot_label in range(1, labeled_shoots.max() + 1):
        shoot_mask = labeled_shoots == shoot_label
        color = colors[(shoot_label - 1) % len(colors)]
        vis_img[shoot_mask] = color
    
    # Draw roots with matching shoot colors (or white for unmatched)
    for root_label in structures['unique_labels']:
        root_skeleton = structures['labeled_skeleton'] == root_label
        
        # Thicken the skeleton for visibility
        if root_thickness > 0:
            root_skeleton = ndimage.binary_dilation(
                root_skeleton, 
                structure=ndimage.generate_binary_structure(2, 1),
                iterations=root_thickness
            )
        
        match = root_matches.get(root_label)
        if match and match is not None:
            shoot_label = match['shoot_label']
            color = colors[(shoot_label - 1) % len(colors)]
        else:
            color = [128, 128, 128]  # Gray for unmatched
        
        vis_img[root_skeleton] = color
    
    # Find bounding box if zooming
    if zoom_to_content:
        # Find all non-black pixels
        content_mask = np.any(vis_img > 0, axis=2)
        rows, cols = np.where(content_mask)
        
        if len(rows) > 0:
            y_min = max(0, rows.min() - padding)
            y_max = min(h, rows.max() + padding)
            x_min = max(0, cols.min() - padding)
            x_max = min(w, cols.max() + padding)
            
            vis_img = vis_img[y_min:y_max, x_min:x_max]
    
    # Plot
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(vis_img)
    ax.set_title('Root-Shoot Matches (same color = matched)', fontsize=14)
    ax.axis('off')
    
    # Create legend
    legend_elements = []
    matched_shoots = set()
    for root_label, match in root_matches.items():
        if match and match is not None:
            matched_shoots.add(match['shoot_label'])
    
    for shoot_label in sorted(matched_shoots):
        color = np.array(colors[(shoot_label - 1) % len(colors)]) / 255.0
        legend_elements.append(
            mpatches.Patch(color=color, label=f'Shoot/Root {shoot_label}')
        )
    
    # Add unmatched if any
    unmatched_count = sum(1 for m in root_matches.values() if m is None)
    if unmatched_count > 0:
        legend_elements.append(
            mpatches.Patch(color=[0.5, 0.5, 0.5], 
                          label=f'Unmatched ({unmatched_count})')
        )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\nMatch Summary:")
    print(f"Total root structures: {len(structures['unique_labels'])}")
    print(f"Matched: {sum(1 for m in root_matches.values() if m is not None)}")
    print(f"Unmatched (noise): {unmatched_count}")


def get_skeleton_path_from_branch(skeleton, branch_row):
    """
    Extract actual skeleton pixels for a branch using the labeled skeleton.
    
    Args:
        skeleton: Binary skeleton array
        branch_row: Row from branch_data DataFrame
        
    Returns:
        Array of (row, col) coordinates along the skeleton path
    """
    from skimage.morphology import skeletonize
    from skimage.graph import route_through_array
    
    start_coord = (int(branch_row['image-coord-src-0']), 
                   int(branch_row['image-coord-src-1']))
    end_coord = (int(branch_row['image-coord-dst-0']), 
                 int(branch_row['image-coord-dst-1']))
    
    # Create cost array: low cost on skeleton, high elsewhere
    cost = np.where(skeleton, 1, 10000)
    
    try:
        indices, weight = route_through_array(
            cost, start_coord, end_coord, fully_connected=True
        )
        return np.array(indices)
    except:
        # Fallback to straight line
        return np.array([start_coord, end_coord])

def visualize_single_root_structure(skeleton, branch_data, title='Root Structure',
                                   dilate_iterations=2, zoom_to_content=True, 
                                   padding=50, trunk_path=None,
                                   path_edge_width=3, path_node_size=10,
                                   show_edge_dots=True, edge_dot_spacing=50,
                                   show_cumulative_length=True,
                                   output_path=None):
    """Visualize a single root structure with trunk path and cumulative length measurements.
    
    Displays a root skeleton with the main trunk path highlighted in blue, showing nodes
    along the path with gradient coloring (red to yellow) and optional cumulative length
    labels at regular intervals. The path follows the actual skeleton pixels rather than
    straight-line connections between nodes.
    
    Args:
        skeleton (np.ndarray): Binary skeleton array (H, W) for a single root structure.
        branch_data (pd.DataFrame): DataFrame containing branch information with columns:
            'node-id-src', 'node-id-dst', 'image-coord-src-0', 'image-coord-src-1',
            'image-coord-dst-0', 'image-coord-dst-1', 'branch-distance'.
        title (str, optional): Title for the visualization. Defaults to 'Root Structure'.
        dilate_iterations (int, optional): Number of morphological dilation iterations 
            to thicken skeleton for visibility. Defaults to 2.
        zoom_to_content (bool, optional): Whether to zoom to the bounding box of the 
            root structure. Defaults to True.
        padding (int, optional): Padding in pixels around the zoomed content. 
            Defaults to 50.
        trunk_path (list, optional): List of tuples from find_farthest_endpoint_path:
            [(node_id, branch_idx, distance, vertical), ...]. If None, only skeleton
            is shown. Defaults to None.
        path_edge_width (float, optional): Line width for trunk path edges. 
            Defaults to 3.
        path_node_size (float, optional): Marker size for nodes along trunk path. 
            Defaults to 10.
        show_edge_dots (bool, optional): Whether to show cyan dots along the trunk path.
            Defaults to True.
        edge_dot_spacing (int, optional): Spacing in pixels between edge dots. Minimum
            spacing is 50 pixels to prevent overcrowding. Defaults to 50.
        show_cumulative_length (bool, optional): Whether to display yellow labels showing
            cumulative length at regular intervals along the path. Defaults to True.
        output_path (str or Path): path to save image
    
    Returns:
        None: Displays matplotlib figure showing the root structure visualization.
    
    Notes:
        - Trunk path is drawn in blue following actual skeleton pixels
        - Cyan dots mark regular intervals along the path
        - Nodes are colored with autumn colormap (red=start, yellow=end)
        - Yellow labels show cumulative length every other dot
        - Lime green label shows total length at the final node
        - Node IDs are displayed in white text boxes offset from nodes
    
    Examples:
        >>> skeleton = structures['labeled_skeleton'] == root_label
        >>> branch_data = structures['roots'][root_label]['branch_data']
        >>> path = find_farthest_endpoint_path(branch_data, top_node, direction='down')
        >>> visualize_single_root_structure(
        ...     skeleton, branch_data, 
        ...     title='Root 35 Structure',
        ...     trunk_path=path,
        ...     edge_dot_spacing=100
        ... )
    """
    from scipy.ndimage import binary_dilation
    
    # Thicken skeleton for display
    thick_skeleton = binary_dilation(skeleton, iterations=dilate_iterations)
    skeleton_uint8 = (thick_skeleton * 255).astype(np.uint8)
    
    # Get all node coordinates
    all_node_coords = {}
    for idx, row in branch_data.iterrows():
        all_node_coords[row['node-id-src']] = (row['image-coord-src-0'], 
                                                row['image-coord-src-1'])
        all_node_coords[row['node-id-dst']] = (row['image-coord-dst-0'], 
                                                row['image-coord-dst-1'])
    
    # Get trunk path nodes
    path_nodes = []
    path_branch_indices = []
    if trunk_path:
        for node, branch_idx, _, _ in trunk_path:
            path_nodes.append(int(node))
            if branch_idx is not None:
                path_branch_indices.append(branch_idx)
    
    # Bounding box
    if zoom_to_content and all_node_coords:
        rows = [coord[0] for coord in all_node_coords.values()]
        cols = [coord[1] for coord in all_node_coords.values()]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
    else:
        zoom_to_content = False
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(skeleton_uint8, cmap='gray')
    
    # Draw trunk path and collect cumulative lengths
    all_path_pixels = []
    cumulative_lengths = []  # Store (pixel_index, cumulative_length)
    current_length = 0.0
    total_length = 0.0
    
    if trunk_path:
        for i, (node, branch_idx, distance, _) in enumerate(trunk_path):
            if branch_idx is not None:
                branch_row = branch_data.iloc[branch_idx]
                
                # Get actual skeleton pixels for this branch
                path_pixels = get_skeleton_path_from_branch(skeleton, branch_row)
                
                # Track cumulative length at start of this segment
                pixel_start_idx = len(np.vstack(all_path_pixels)) if all_path_pixels else 0
                cumulative_lengths.append((pixel_start_idx, current_length))
                
                all_path_pixels.append(path_pixels)
                
                # Add this segment's length
                current_length += distance
                total_length += distance
                
                # Draw the path in blue
                ax.plot(path_pixels[:, 1], path_pixels[:, 0], '-',
                       color='blue', linewidth=path_edge_width,
                       alpha=0.8, zorder=2)
        
        # Add final length
        if all_path_pixels:
            final_idx = len(np.vstack(all_path_pixels))
            cumulative_lengths.append((final_idx, current_length))
        
        # Draw dots and labels
        if show_edge_dots and all_path_pixels:
            # Concatenate all segments
            combined_path = np.vstack(all_path_pixels)
            
            # Ensure minimum spacing of 50px
            actual_spacing = max(edge_dot_spacing, 50)
            sampled_indices = np.arange(0, len(combined_path), actual_spacing)
            
            # Draw dots
            sampled_pixels = combined_path[sampled_indices]
            ax.plot(sampled_pixels[:, 1], sampled_pixels[:, 0], 'o',
                   color='cyan', markersize=4, alpha=0.9, zorder=2.5,
                   markeredgecolor='blue', markeredgewidth=0.5)
            
            # Add cumulative length labels
            if show_cumulative_length:
                for sample_idx in sampled_indices[::2]:  # Label every other dot to reduce clutter
                    if sample_idx < len(combined_path):
                        # Find cumulative length at this pixel
                        cum_length = 0.0
                        for seg_idx, (pixel_idx, length) in enumerate(cumulative_lengths[:-1]):
                            next_pixel_idx = cumulative_lengths[seg_idx + 1][0]
                            if pixel_idx <= sample_idx < next_pixel_idx:
                                # Interpolate within this segment
                                segment_progress = (sample_idx - pixel_idx) / max(next_pixel_idx - pixel_idx, 1)
                                next_length = cumulative_lengths[seg_idx + 1][1]
                                cum_length = length + segment_progress * (next_length - length)
                                break
                        else:
                            cum_length = current_length
                        
                        row, col = combined_path[sample_idx]
                        ax.text(col + 15, row, f'{cum_length:.0f}px',
                               color='yellow', fontsize=8, fontweight='bold',
                               ha='left', va='center',
                               bbox=dict(boxstyle='round,pad=0.2',
                                       facecolor='black', alpha=0.6),
                               zorder=4)
    
    # Draw path nodes
    if trunk_path:
        for i, node_id in enumerate(path_nodes):
            if node_id in all_node_coords:
                row, col = all_node_coords[node_id]
                color = plt.cm.autumn(i / max(len(path_nodes)-1, 1))
                ax.plot(col, row, 'o', color=color, markersize=path_node_size,
                       zorder=3, markeredgecolor='white', markeredgewidth=1.5)
                
                # Add node label
                offset_x = 20
                offset_y = -5
                ax.text(col + offset_x, row + offset_y, str(node_id),
                       color='white', fontsize=9, fontweight='bold',
                       ha='left', va='center',
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='black', alpha=0.7),
                       zorder=4)
                
                # Add total length at final node
                if i == len(path_nodes) - 1 and show_cumulative_length:
                    ax.text(col + 15, row + 20, f'{total_length:.1f}px',
                           color='lime', fontsize=10, fontweight='bold',
                           ha='left', va='top',
                           bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='black', alpha=0.8),
                           zorder=5)
    
    if zoom_to_content:
        ax.set_xlim(col_min - padding, col_max + padding)
        ax.set_ylim(row_max + padding, row_min - padding)
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    if output_path:
        plt.savefig(str(output_path))
    plt.show()


def visualize_root_lengths(structures, top_node_results, labeled_shoots, 
                          zoom_to_content=True, padding=50,
                          show_detailed_roots=False, detail_viz_kwargs=None):
    """Visualize matched roots with their calculated lengths displayed.
    
    Creates an overview visualization showing all matched root structures color-coded
    by their associated shoot regions, with length measurements labeled. Optionally
    displays detailed individual root visualizations below the overview.
    
    Args:
        structures (dict): Dictionary from extract_root_structures containing:
            - 'labeled_skeleton': Array with labeled skeleton structures
            - 'roots': Dictionary of root data by label
        top_node_results (dict): Dictionary from find_top_nodes_from_shoot with keys:
            - root_label: Dict containing 'branch_data', 'shoot_label', 'top_nodes'
        labeled_shoots (np.ndarray): Labeled array from label_shoot_regions where each
            shoot region has a unique integer label.
        zoom_to_content (bool, optional): If True, zoom to bounding box of all content.
            Defaults to True.
        padding (int, optional): Pixels to add around bounding box when zooming.
            Defaults to 50.
        show_detailed_roots (bool, optional): If True, display detailed visualization
            for each individual root below the overview. Defaults to False.
        detail_viz_kwargs (dict, optional): Keyword arguments to pass to 
            visualize_single_root_structure for detailed views. Common options:
            - 'dilate_iterations': int, skeleton thickness (default: 2)
            - 'path_edge_width': float, trunk path line width (default: 3)
            - 'path_node_size': float, node marker size (default: 10)
            - 'show_edge_dots': bool, show dots along path (default: True)
            - 'edge_dot_spacing': int, spacing between dots (default: 50)
            - 'show_cumulative_length': bool, show length labels (default: True)
            If None, uses default values. Defaults to None.
    
    Returns:
        None: Displays matplotlib figures showing root length visualizations.
    
    Notes:
        - Overview uses color-coding: each root matches its shoot color
        - Roots are thickened with binary dilation for visibility
        - Length labels appear at root centroids in white text boxes
        - Detailed views are indexed (idx=0, idx=1, etc.) in display order
        - Failed root measurements appear in gray with no length label
    
    Examples:
        >>> # Basic usage - overview only
        >>> visualize_root_lengths(structures, top_node_results, labeled_shoots)
        
        >>> # With detailed individual visualizations
        >>> visualize_root_lengths(
        ...     structures, top_node_results, labeled_shoots,
        ...     show_detailed_roots=True,
        ...     detail_viz_kwargs={
        ...         'edge_dot_spacing': 100,
        ...         'path_edge_width': 2,
        ...         'show_cumulative_length': True
        ...     }
        ... )
    """
    
    # Default kwargs for detailed visualizations
    if detail_viz_kwargs is None:
        detail_viz_kwargs = {}
    
    # Create RGB image
    h, w = structures['labeled_skeleton'].shape
    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Color map
    colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [255, 0, 255],    # Magenta
        [0, 255, 255],    # Cyan
        [255, 128, 0],    # Orange
        [128, 0, 255],    # Purple
    ]
    
    # Draw shoots
    for shoot_label in range(1, labeled_shoots.max() + 1):
        shoot_mask = labeled_shoots == shoot_label
        color = colors[(shoot_label - 1) % len(colors)]
        vis_img[shoot_mask] = color
    
    # Process and draw roots with paths
    root_info = []
    root_details = []  # Store data for detailed visualizations
    
    for root_label, result in top_node_results.items():
        branch_data = result['branch_data']
        shoot_label = result['shoot_label']
        top_node = result['top_nodes'][0][0]
        
        # Get root skeleton and thicken
        root_skeleton = structures['labeled_skeleton'] == root_label
        root_skeleton = ndimage.binary_dilation(
            root_skeleton, 
            structure=ndimage.generate_binary_structure(2, 1),
            iterations=3
        )
        
        color = colors[(shoot_label - 1) % len(colors)]
        
        try:
            # Find path and calculate length
            path = find_farthest_endpoint_path(
                branch_data, top_node, 
                direction='down', use_smart_scoring=True,
                verbose=False
            )
            root_length = calculate_skeleton_length_px(path)
            
            # Draw root
            vis_img[root_skeleton] = color
            
            # Store info for labels
            y_coords, x_coords = np.where(root_skeleton)
            centroid_y, centroid_x = np.mean(y_coords), np.mean(x_coords)
            root_info.append((centroid_x, centroid_y, root_label, shoot_label, root_length))
            
            # Store for detailed visualization
            if show_detailed_roots:
                root_details.append({
                    'root_label': root_label,
                    'skeleton': structures['labeled_skeleton'] == root_label,
                    'branch_data': branch_data,
                    'path': path,
                    'shoot_label': shoot_label
                })
            
        except Exception as e:
            # Draw root in gray if failed
            vis_img[root_skeleton] = [128, 128, 128]
    
    # Zoom if requested
    if zoom_to_content:
        content_mask = np.any(vis_img > 0, axis=2)
        rows, cols = np.where(content_mask)
        if len(rows) > 0:
            y_min = max(0, rows.min() - padding)
            y_max = min(h, rows.max() + padding)
            x_min = max(0, cols.min() - padding)
            x_max = min(w, cols.max() + padding)
            vis_img = vis_img[y_min:y_max, x_min:x_max]
            # Adjust coordinates for zoom
            root_info = [(x - x_min, y - y_min, rl, sl, length) 
                        for x, y, rl, sl, length in root_info]
    
    # Plot overview
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(vis_img)
    
    # Add text labels
    for x, y, root_label, shoot_label, length in root_info:
        ax.text(x, y, f'{length:.1f}px', 
               color='white', fontsize=10, weight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_title('Root Lengths (pixels)', fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Display detailed visualizations if requested
    if show_detailed_roots and root_details:
        print(f"\n{'='*60}")
        print(f"Detailed Root Visualizations ({len(root_details)} roots)")
        print(f"{'='*60}\n")
        
        for idx, detail in enumerate(root_details):
            print(f"Displaying Root {detail['root_label']} (idx={idx})...")
            
            visualize_single_root_structure(
                skeleton=detail['skeleton'],
                branch_data=detail['branch_data'],
                title=f"Root {detail['root_label']} (idx={idx}) - Shoot {detail['shoot_label']}",
                trunk_path=detail['path'],
                **detail_viz_kwargs
            )