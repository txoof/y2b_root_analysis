import numpy as np
import pandas as pd
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from skimage.morphology import skeletonize
from skimage.measure import label
from scipy.spatial.distance import cdist
from scipy import ndimage
from pathlib import Path


# import matplotlib.pyplot as plt

def find_most_vertical_branch(branch_data, current_node, direction='up', 
                               verticality_weight=0.7, height_weight=0.3):
    """
    From current_node, find the branch that balances verticality AND height gained.
    
    Args:
        branch_data: DataFrame with branch information
        current_node: The node we're starting from
        direction: 'up' or 'down'
        verticality_weight: Weight for slope (0-1)
        height_weight: Weight for vertical distance gained (0-1)
    
    Returns:
        tuple: (next_node, branch_index, slope, vertical_change) or (None, None, None, None)
    """
    connected_branches = branch_data[
        (branch_data['node-id-src'] == current_node) | 
        (branch_data['node-id-dst'] == current_node)
    ]
    
    best_score = 0
    best_next_node = None
    best_branch_idx = None
    best_slope = None
    best_vertical = None
    
    candidates = []
    
    for idx, row in connected_branches.iterrows():
        if row['node-id-src'] == current_node:
            next_node = row['node-id-dst']
            vertical_change = row['coord-src-0'] - row['coord-dst-0']
            horizontal_change = abs(row['coord-src-1'] - row['coord-dst-1'])
        else:
            next_node = row['node-id-src']
            vertical_change = row['coord-dst-0'] - row['coord-src-0']
            horizontal_change = abs(row['coord-dst-1'] - row['coord-src-1'])
        
        # Check direction
        if direction == 'up':
            condition = vertical_change > 0
        elif direction == 'down':
            condition = vertical_change < 0
        else:
            raise ValueError("direction must be 'up' or 'down'")
        
        if condition:
            # Calculate slope
            if horizontal_change > 0:
                slope = abs(vertical_change) / horizontal_change
            else:
                slope = float('inf')
            
            # Normalize scores (slope can be 0-inf, vertical_change is in pixels)
            # Use ratio to max for normalization
            candidates.append({
                'idx': idx,
                'next_node': next_node,
                'slope': slope,
                'vertical_change': abs(vertical_change),
                'horizontal_change': horizontal_change
            })
    
    if not candidates:
        return None, None, None, None
    
    # Normalize scores
    max_slope = max(c['slope'] if c['slope'] != float('inf') else 0 for c in candidates)
    max_vertical = max(c['vertical_change'] for c in candidates)
    
    # Calculate combined scores
    for c in candidates:
        slope_norm = (c['slope'] if c['slope'] != float('inf') else max_slope * 2) / (max_slope if max_slope > 0 else 1)
        vertical_norm = c['vertical_change'] / (max_vertical if max_vertical > 0 else 1)
        
        combined_score = verticality_weight * slope_norm + height_weight * vertical_norm
        
        print(f"    B{c['idx']} → {int(c['next_node'])}: slope={c['slope']:.2f}, vert={c['vertical_change']:.1f}, score={combined_score:.3f}")
        
        if combined_score > best_score:
            best_score = combined_score
            best_next_node = c['next_node']
            best_branch_idx = c['idx']
            best_slope = c['slope']
            best_vertical = c['vertical_change']
    
    return best_next_node, best_branch_idx, best_slope, best_vertical


def trace_vertical_path(branch_data, start_node, direction='down', 
                       verticality_weight=0.5, height_weight=0.5, max_steps=100):
    """Trace the most vertical path from start_node."""
    path = []
    current_node = start_node
    visited_nodes = set()
    
    for step in range(max_steps):
        if current_node in visited_nodes:
            print(f"  Loop detected at node {int(current_node)}, stopping")
            break
        visited_nodes.add(current_node)
        
        # Now expects 4 return values
        next_node, branch_idx, slope, vertical = find_most_vertical_branch(
            branch_data, current_node, direction, verticality_weight, height_weight
        )
        
        if next_node is None:
            print(f"  Reached endpoint at node {int(current_node)}")
            break
        
        path.append((current_node, branch_idx, slope, vertical))
        print(f"  Step {step}: Node {int(current_node)} → {int(next_node)} via B{branch_idx}")
        
        current_node = next_node
    
    path.append((current_node, None, None, None))
    return path



def label_skeleton(binary_mask):
    """
    Take a binary mask, skeletonize it, and return labeled connected components.
    
    Args:
        binary_mask: Binary numpy array (bool or 0/255)
    
    Returns:
        tuple: (skeleton, labeled_skeleton, unique_labels)
            - skeleton: Binary skeleton array
            - labeled_skeleton: Labeled skeleton with unique IDs for each component
            - unique_labels: Array of unique label IDs (excluding background 0)
    """
    
    # Ensure binary
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)
    
    # Skeletonize
    skeleton = skeletonize(binary_mask)
    
    # Label connected components
    labeled_skeleton = label(skeleton)
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(labeled_skeleton)
    unique_labels = unique_labels[unique_labels != 0]
    
    return skeleton, labeled_skeleton, unique_labels

def find_top_nodes(branch_data, n_nodes=1, threshold=None):
    """
    Find the node(s) closest to the top of the image (lowest row values).
    
    Args:
        branch_data: DataFrame with branch information
        n_nodes: Number of top nodes to return
        threshold: Optional - return all nodes within this many pixels of the top
    
    Returns:
        list of node IDs at the top
    """
    # Get all unique nodes and their coordinates
    node_coords = {}
    for idx, row in branch_data.iterrows():
        node_coords[row['node-id-src']] = row['coord-src-0']
        node_coords[row['node-id-dst']] = row['coord-dst-0']
    
    # Sort by row coordinate (lowest = top)
    sorted_nodes = sorted(node_coords.items(), key=lambda x: x[1])
    
    if threshold is not None:
        # Return all nodes within threshold pixels of the top
        min_row = sorted_nodes[0][1]
        top_nodes = [node_id for node_id, row in sorted_nodes if row <= min_row + threshold]
        print(f"Top node at row {min_row:.0f}")
        print(f"Found {len(top_nodes)} nodes within {threshold} pixels of top")
    else:
        # Return top n nodes
        top_nodes = [node_id for node_id, row in sorted_nodes[:n_nodes]]
        print(f"Top {n_nodes} node(s):")
        for node_id, row in sorted_nodes[:n_nodes]:
            print(f"  Node {int(node_id)} at row {row:.0f}")
    
    return top_nodes

def extract_root_structures(binary_mask, verbose=False):
    """
    Analyze all root structures in a binary mask.
    
    Args:
        binary_mask: Binary numpy array (bool or 0/255) or path-like object to image file
    
    Returns:
        dict: {
            'skeleton': full skeleton array,
            'labeled_skeleton': labeled skeleton array,
            'unique_labels': array of label IDs,
            'roots': {
                label_id: {
                    'mask': binary mask for this root only,
                    'skeleton': skeleton for this root only,
                    'branch_data': DataFrame with branch information,
                    'num_nodes': number of nodes,
                    'num_branches': number of branches,
                    'total_pixels': total skeleton pixels
                },
                ...
            }
        }
    """
    # Load from file if path provided
    if not isinstance(binary_mask, np.ndarray):
        binary_mask = cv2.imread(str(binary_mask), cv2.IMREAD_GRAYSCALE)
    
    # Ensure binary
    if binary_mask.dtype != bool:
        binary_mask = binary_mask.astype(bool)
    
    # Get labeled skeleton
    skeleton, labeled_skeleton, unique_labels = label_skeleton(binary_mask)
    
    # Initialize results
    results = {
        'skeleton': skeleton,
        'labeled_skeleton': labeled_skeleton,
        'unique_labels': unique_labels,
        'roots': {}
    }
    if verbose:
        print(f"Analyzing {len(unique_labels)} root structure(s)...")
    
    # Analyze each root separately
    for label_id in unique_labels:
        if verbose:
            print(f"\n--- Root {int(label_id)} ---")
        
        # Extract this root's skeleton
        single_root_mask = (labeled_skeleton == label_id)
        
        # Get branch data for this root
        try:
            skeleton_obj = Skeleton(single_root_mask)
            branch_data = summarize(skeleton_obj)
            
            # Get node count
            unique_nodes = set(branch_data['node-id-src']).union(
                set(branch_data['node-id-dst']))
            num_nodes = len(unique_nodes)
            num_branches = len(branch_data)
            total_pixels = np.sum(single_root_mask)
            
            # Store results
            results['roots'][int(label_id)] = {
                'mask': single_root_mask,
                'skeleton': single_root_mask,
                'branch_data': branch_data,
                'num_nodes': num_nodes,
                'num_branches': num_branches,
                'total_pixels': total_pixels
            }
            if verbose:
                print(f"  Nodes: {num_nodes}, Branches: {num_branches}, Pixels: {total_pixels}")
            
        except Exception as e:
            if verbose:
                print(f"  ERROR analyzing root {int(label_id)}: {e}")
            results['roots'][int(label_id)] = {
                'mask': single_root_mask,
                'skeleton': single_root_mask,
                'branch_data': None,
                'num_nodes': 0,
                'num_branches': 0,
                'total_pixels': np.sum(single_root_mask),
                'error': str(e)
            }

    if verbose:    
        print(f"\n=== Summary ===")
        print(f"Total roots analyzed: {len(results['roots'])}")
        successful = sum(1 for r in results['roots'].values() if r['branch_data'] is not None)
        print(f"Successfully analyzed: {successful}")
    
    return results

def find_farthest_endpoint_path(branch_data, start_node, direction='down', 
                               use_smart_scoring=False, 
                               horizontal_penalty=0.5,
                               straightness_weight=1.0,
                               verbose=True):
    """
    Find the path to the endpoint that is farthest away in straight-line distance.
    
    Args:
        branch_data: DataFrame with branch information
        start_node: Node to start from
        direction: 'down' or 'up' (for filtering endpoints)
        use_smart_scoring: If True, use scoring that penalizes horizontal deviation and rewards straightness
        horizontal_penalty: Weight for horizontal deviation penalty (higher = more penalty)
        straightness_weight: Weight for straightness ratio (higher = more reward for straight paths)
        verbose: Print progress
    
    Returns:
        Best path: [(node, branch_idx, ...), ...]
    """
    import networkx as nx
    
    # Build NetworkX graph
    G = nx.Graph()
    
    # Get node coordinates
    node_coords = {}
    for idx, row in branch_data.iterrows():
        src = row['node-id-src']
        dst = row['node-id-dst']
        
        # Store coordinates
        node_coords[src] = (row['coord-src-0'], row['coord-src-1'])
        node_coords[dst] = (row['coord-dst-0'], row['coord-dst-1'])
        
        # Add edge
        G.add_edge(src, dst, branch_idx=idx, distance=row['branch-distance'])
    
    # Get start node coordinates
    if start_node not in node_coords:
        print(f"Error: Start node {int(start_node)} not found in graph")
        return [(start_node, None, None, None)]
    
    start_row, start_col = node_coords[start_node]
    
    if verbose:
        print(f"Start node {int(start_node)} at (row={start_row:.0f}, col={start_col:.0f})")
        print(f"Scoring mode: {'SMART (considering straightness & horizontal deviation)' if use_smart_scoring else 'SIMPLE (just Euclidean distance)'}")
        
    # Find all endpoints (degree 1 nodes)
    endpoints = [node for node in G.nodes() if G.degree(node) == 1]
    endpoints = [ep for ep in endpoints if ep != start_node]
    
    if verbose:
        print(f"Found {len(endpoints)} candidate endpoints")
    
    # Evaluate each endpoint
    endpoint_scores = []
    
    for endpoint in endpoints:
        end_row, end_col = node_coords[endpoint]
        
        # Calculate basic distances
        euclidean_dist = np.sqrt((end_row - start_row)**2 + (end_col - start_col)**2)
        vertical_dist = end_row - start_row  # Positive = going down
        horizontal_dist = abs(end_col - start_col)  # Horizontal deviation
        
        # Filter by direction
        if direction == 'down' and vertical_dist <= 0:
            continue
        elif direction == 'up' and vertical_dist >= 0:
            continue
        
        # Find a path to calculate straightness
        try:
            node_path = nx.shortest_path(G, start_node, endpoint, weight='distance')
            
            # Calculate actual skeleton path length
            skeleton_length = 0
            for i in range(len(node_path) - 1):
                edge_data = G[node_path[i]][node_path[i + 1]]
                skeleton_length += edge_data['distance']
            
            # Calculate straightness ratio
            # 1.0 = perfectly straight, <1.0 = wiggly
            straightness = euclidean_dist / skeleton_length if skeleton_length > 0 else 0
            
        except nx.NetworkXNoPath:
            continue
        
        # Calculate score based on mode
        if use_smart_scoring:
            # SMART SCORING: vertical distance * straightness - horizontal penalty
            # This rewards:
            #   - Going far down (vertical_dist)
            #   - Straight paths (straightness)
            # This penalizes:
            #   - Horizontal deviation (horizontal_dist)
            score = (abs(vertical_dist) * straightness_weight * straightness 
                    - horizontal_penalty * horizontal_dist)
        else:
            # SIMPLE SCORING: just use Euclidean distance
            score = euclidean_dist
        
        endpoint_scores.append({
            'endpoint': endpoint,
            'score': score,
            'euclidean_dist': euclidean_dist,
            'vertical_dist': abs(vertical_dist),
            'horizontal_dist': horizontal_dist,
            'straightness': straightness,
            'skeleton_length': skeleton_length,
            'coords': (end_row, end_col)
        })
    
    if not endpoint_scores:
        print("No valid endpoints found in specified direction!")
        return [(start_node, None, None, None)]
    
    # Sort by score (highest first)
    endpoint_scores.sort(key=lambda x: x['score'], reverse=True)
    
    if verbose:
        print(f"\nTop 5 candidates:")
        for i, ep_info in enumerate(endpoint_scores[:5]):
            print(f"  {i+1}. Node {int(ep_info['endpoint'])}: "
                  f"score={ep_info['score']:.1f}, "
                  f"euclidean={ep_info['euclidean_dist']:.1f}px, "
                  f"vertical={ep_info['vertical_dist']:.1f}px, "
                  f"horizontal={ep_info['horizontal_dist']:.1f}px, "
                  f"straightness={ep_info['straightness']:.2f}")
    
    # Get the best endpoint
    best = endpoint_scores[0]
    target_endpoint = best['endpoint']
    
    if verbose:
        print(f"\nSelected: Node {int(target_endpoint)} (score: {best['score']:.1f})")
    
    # Find shortest path to target
    try:
        node_path = nx.shortest_path(G, start_node, target_endpoint, weight='distance')
    except nx.NetworkXNoPath:
        print(f"No path found to endpoint {int(target_endpoint)}")
        return [(start_node, None, None, None)]
    
    # Convert to detailed path
    detailed_path = []
    total_skeleton_length = 0
    
    for i in range(len(node_path) - 1):
        current = node_path[i]
        next_node = node_path[i + 1]
        
        edge_data = G[current][next_node]
        branch_idx = edge_data['branch_idx']
        distance = edge_data['distance']
        
        total_skeleton_length += distance
        
        curr_row, curr_col = node_coords[current]
        next_row, next_col = node_coords[next_node]
        vertical_dist = next_row - curr_row
        
        detailed_path.append((current, branch_idx, distance, vertical_dist))
    
    detailed_path.append((node_path[-1], None, None, None))
    
    if verbose:
        node_sequence = [int(node) for node, _, _, _ in detailed_path]
        print(f"\nPath: {' → '.join(map(str, node_sequence[:15]))}")
        if len(node_sequence) > 15:
            print(f"  ... (total {len(node_sequence)} nodes)")
        print(f"Metrics:")
        print(f"  Euclidean: {best['euclidean_dist']:.1f}px")
        print(f"  Skeleton: {total_skeleton_length:.1f}px")
        print(f"  Straightness: {best['straightness']:.2f}")
        print(f"  Horizontal deviation: {best['horizontal_dist']:.1f}px")
    
    return detailed_path


def calculate_skeleton_length_px(detailed_path):
    """
    Calculate the total skeleton path length in pixels.
    
    Args:
        detailed_path: List of tuples (node, branch_idx, distance, vertical_dist)
                      from find_farthest_endpoint_path()
    
    Returns:
        float: Total path length in pixels
    """
    total_length = 0.0
    
    # Sum all distance values, skipping the last element which has None for distance
    for node, branch_idx, distance, vertical_dist in detailed_path:
        if distance is not None:
            total_length += distance
    
    return total_length






def process_matched_roots_to_lengths(structures, top_node_results, labeled_shoots, num_shoots):
    """
    Process matched roots and return array of lengths ordered by shoot position (left to right).
    
    Args:
        structures: Dict from extract_root_structures
        top_node_results: Dict from find_top_nodes_from_shoot
        labeled_shoots: Labeled shoot array
        num_shoots: Number of shoots (typically 5)
        
    Returns:
        np.array: Array of root lengths in pixels, ordered by shoot x-position (left to right).
                 Zero-indexed where [0] is leftmost shoot. Returns 0.0 for shoots without matched roots.
    """
    
    
    # Get x-position (centroid) of each shoot for left-to-right ordering
    shoot_positions = {}
    for shoot_label in range(1, num_shoots + 1):
        shoot_mask = labeled_shoots == shoot_label
        y_coords, x_coords = np.where(shoot_mask)
        if len(x_coords) > 0:
            centroid_x = np.mean(x_coords)
            shoot_positions[shoot_label] = centroid_x
        else:
            shoot_positions[shoot_label] = float('inf')
    
    # Sort shoots by x-position (left to right)
    sorted_shoots = sorted(shoot_positions.items(), key=lambda x: x[1])
    shoot_order = [label for label, _ in sorted_shoots]
    
    # Create mapping from shoot_label to root_length
    shoot_to_length = {}
    
    for root_label, result in top_node_results.items():
        branch_data = result['branch_data']
        shoot_label = result['shoot_label']
        top_node = result['top_nodes'][0][0]
        
        try:
            # Find longest path from top node
            path = find_farthest_endpoint_path(
                branch_data, 
                top_node, 
                direction='down', 
                use_smart_scoring=True,
                verbose=False
            )
            
            # Calculate length
            root_length = calculate_skeleton_length_px(path)
            
            # Store length for this shoot
            shoot_to_length[shoot_label] = root_length
            
        except Exception as e:
            print(f"Warning: Failed to process root {root_label} for shoot {shoot_label}: {e}")
            shoot_to_length[shoot_label] = 0.0
    
    # Build output array ordered by shoot position (left to right)
    lengths_array = np.array([
        shoot_to_length.get(shoot_label, 0.0) 
        for shoot_label in shoot_order
    ])
    
    return lengths_array