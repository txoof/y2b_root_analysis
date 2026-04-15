# library/tf_generators.py
"""TensorFlow data generators for patch-based training."""

import tensorflow as tf
from pathlib import Path
import json


def create_patch_dataset(patch_dir, dataset_type='train', mask_type='root', 
                        batch_size=16, shuffle=True, seed=42, augment=False,
                        augment_config=None):
    """Create a tf.data.Dataset for loading image and mask patches.
    
    Args:
        patch_dir: Directory containing saved patches.
        dataset_type: Either 'train' or 'val'. Default is 'train'.
        mask_type: Which mask to load ('root', 'shoot', or 'seed'). Default is 'root'.
        batch_size: Batch size for training. Default is 16.
        shuffle: Whether to shuffle the dataset. Default is True.
        seed: Random seed for reproducibility. Default is 42.
        augment: Whether to apply data augmentation. Default is False.
        augment_config: Dictionary of augmentation settings. If None, uses defaults.
            Options:
            - 'flip_left_right': bool, default True
            - 'flip_up_down': bool, default True
            - 'rotate': bool, default True (90 degree rotations only)
            - 'brightness': float, max delta for brightness adjustment, default 0.0
            - 'contrast': tuple, (lower, upper) contrast factor range, default (1.0, 1.0)
    
    Returns:
        tf.data.Dataset yielding (image_batch, mask_batch) tuples.
    
    Example:
        >>> config = {'flip_left_right': True, 'rotate': True, 'brightness': 0.1}
        >>> dataset = create_patch_dataset('../../data/patched', 'train', 
        ...                                augment=True, augment_config=config)
    """
    patch_dir = Path(patch_dir)
    
    # Default augmentation config
    if augment_config is None:
        augment_config = {
            'flip_left_right': True,
            'flip_up_down': True,
            'rotate': True,
            'brightness': 0.0,
            'contrast': (1.0, 1.0)
        }
    
    # Get paths to image and mask directories
    image_dir = patch_dir / f'{dataset_type}_images' / dataset_type
    mask_dir = patch_dir / f'{dataset_type}_masks_{mask_type}' / dataset_type
    
    # Get list of image files
    image_files = sorted(list(image_dir.glob('*.png')))
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")
    
    # Create corresponding mask file paths
    mask_files = [mask_dir / img_file.name for img_file in image_files]
    
    # Convert to string paths for TensorFlow
    image_paths = [str(p) for p in image_files]
    mask_paths = [str(p) for p in mask_files]
    
    print(f"Found {len(image_paths)} patches")
    if augment:
        print(f"Augmentation enabled: {augment_config}")
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    
    # Map loading and preprocessing function
    if augment:
        def map_fn(img_path, mask_path):
            return _load_and_augment(img_path, mask_path, augment_config, seed)
        
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            _load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_filtered_patch_dataset(patch_dir, filtered_list_json, dataset_type='train',
                                 batch_size=16, shuffle=True, seed=42, 
                                 augment=False, augment_config=None):
    """Create a tf.data.Dataset using a filtered patch list.
    
    Args:
        patch_dir: Directory containing saved patches.
        filtered_list_json: Path to JSON file with filtered patch list.
        dataset_type: Either 'train' or 'val'. Default is 'train'.
        batch_size: Batch size for training. Default is 16.
        shuffle: Whether to shuffle the dataset. Default is True.
        seed: Random seed for reproducibility. Default is 42.
        augment: Whether to apply data augmentation. Default is False.
        augment_config: Dictionary of augmentation settings (same as before).
    
    Returns:
        tf.data.Dataset yielding (image_batch, mask_batch) tuples.
    
    Example:
        >>> dataset = create_filtered_patch_dataset(
        ...     patch_dir='../../data/patched',
        ...     filtered_list_json='filtered_lists/train_filtered_root.json',
        ...     dataset_type='train',
        ...     augment=True
        ... )
    """
    patch_dir = Path(patch_dir)
    
    # Load filtered patch list
    with open(filtered_list_json, 'r') as f:
        filtered_data = json.load(f)
    
    mask_type = filtered_data['metadata']['mask_type']
    patch_filenames = filtered_data['patch_filenames']
    
    # Default augmentation config
    if augment_config is None:
        augment_config = {
            'flip_left_right': True,
            'flip_up_down': True,
            'rotate': True,
            'brightness': 0.0,
            'contrast': (1.0, 1.0)
        }
    
    # Get paths to image and mask directories
    image_dir = patch_dir / f'{dataset_type}_images' / dataset_type
    mask_dir = patch_dir / f'{dataset_type}_masks_{mask_type}' / dataset_type
    
    # Create full file paths
    image_paths = [str(image_dir / fname) for fname in patch_filenames]
    mask_paths = [str(mask_dir / fname) for fname in patch_filenames]
    
    print(f"Loaded filtered list for {mask_type}")
    print(f"Total patches: {filtered_data['statistics']['total_patches']}")
    print(f"Non-empty: {filtered_data['statistics']['non_empty_patches']}")
    print(f"Empty: {filtered_data['statistics']['empty_patches']}")
    print(f"Empty ratio: {filtered_data['metadata']['actual_empty_ratio']*100:.1f}%")
    
    if augment:
        print(f"Augmentation enabled: {augment_config}")
    
    # Create dataset from file paths
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    # Shuffle if requested
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed)
    
    # Map loading and preprocessing function
    if augment:
        def map_fn(img_path, mask_path):
            return _load_and_augment(img_path, mask_path, augment_config, seed)
        
        dataset = dataset.map(
            map_fn,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        dataset = dataset.map(
            _load_and_preprocess,
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch the dataset
    dataset = dataset.batch(batch_size)
    
    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


def _load_and_augment(image_path, mask_path, augment_config, seed):
    """Load and augment a single image-mask pair.
    
    Args:
        image_path: Path to image file.
        mask_path: Path to mask file.
        augment_config: Dictionary of augmentation settings.
        seed: Random seed.
    
    Returns:
        Tuple of (augmented_image, augmented_mask) tensors.
    """
    # Load image and mask
    image, mask = _load_and_preprocess(image_path, mask_path)
    
    # Random flips
    if augment_config.get('flip_left_right', False):
        do_flip = tf.random.uniform((), seed=seed) > 0.5
        image = tf.cond(do_flip, lambda: tf.image.flip_left_right(image), lambda: image)
        mask = tf.cond(do_flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
    
    if augment_config.get('flip_up_down', False):
        do_flip = tf.random.uniform((), seed=seed) > 0.5
        image = tf.cond(do_flip, lambda: tf.image.flip_up_down(image), lambda: image)
        mask = tf.cond(do_flip, lambda: tf.image.flip_up_down(mask), lambda: mask)
    
    # Random 90 degree rotations (0, 90, 180, 270)
    if augment_config.get('rotate', False):
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32, seed=seed)
        image = tf.image.rot90(image, k=k)
        mask = tf.image.rot90(mask, k=k)
    
    # Color augmentations (only on image, not mask)
    brightness_delta = augment_config.get('brightness', 0.0)
    if brightness_delta > 0:
        image = tf.image.random_brightness(image, brightness_delta, seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    contrast_range = augment_config.get('contrast', (1.0, 1.0))
    if contrast_range[0] != 1.0 or contrast_range[1] != 1.0:
        image = tf.image.random_contrast(image, contrast_range[0], contrast_range[1], seed=seed)
        image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, mask

def _load_and_preprocess(image_path, mask_path):
    """Load and preprocess a single image-mask pair.
    
    Args:
        image_path: Path to image file.
        mask_path: Path to mask file.
    
    Returns:
        Tuple of (image, mask) tensors.
    """
    # Load image
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    
    # Load mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.cast(mask, tf.float32)  # Keep as 0 and 1
    
    return image, mask


def get_content_rich_patches(patch_dir, dataset_type='train', mask_type='root', 
                             n_samples=10, min_mask_coverage=0.01):
    """Get a list of patch filenames that contain meaningful mask content.
    
    Args:
        patch_dir: Directory containing saved patches.
        dataset_type: Either 'train' or 'val'. Default is 'train'.
        mask_type: Which mask type to check. Default is 'root'.
        n_samples: Number of content-rich patches to return. Default is 10.
        min_mask_coverage: Minimum percentage of mask pixels (0-1). Default is 0.01 (1%).
    
    Returns:
        List of patch filenames that have sufficient mask content.
    """
    import cv2
    import random
    
    patch_dir = Path(patch_dir)
    mask_dir = patch_dir / f'{dataset_type}_masks_{mask_type}' / dataset_type
    
    # Get all mask files
    mask_files = list(mask_dir.glob('*.png'))
    
    # Find patches with content
    content_patches = []
    for mask_file in mask_files:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        mask_coverage = (mask > 0).sum() / mask.size
        
        if mask_coverage >= min_mask_coverage:
            content_patches.append(mask_file.name)
    
    print(f"Found {len(content_patches)} patches with >{min_mask_coverage*100:.1f}% mask coverage")
    
    # Return random sample
    return random.sample(content_patches, min(n_samples, len(content_patches)))