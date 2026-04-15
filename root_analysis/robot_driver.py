"""Robot driver module for OT-2 simulation control and computer vision integration.

This module provides the RobotDriver class for controlling the OT-2 robot in simulation
using either PID or RL controllers, and the get_coord_from_plate function for processing
specimen images through the complete computer vision pipeline to extract root tip coordinates.

The RobotDriver handles:
- Moving the robot to target positions
- Dispensing inoculum at root tip locations
- Supporting both PID and RL control strategies
- Settling detection and tolerance checking

The get_coord_from_plate function integrates:
- U-Net inference for shoot and root segmentation
- Shoot mask cleaning and validation
- Root mask edge repair
- Complete root-shoot matching pipeline
- Coordinate conversion to robot frame

Typical Usage:
    >>> from robot_driver import RobotDriver, get_coord_from_plate
    >>> 
    >>> # Get coordinates from specimen image
    >>> df = get_coord_from_plate(
    ...     plate_image_path='path/to/image.png',
    ...     shoot_model='models/shoot_unet.h5',
    ...     root_model='models/root_unet.h5'
    ... )
    >>> 
    >>> # Create robot driver with PID controller
    >>> driver = RobotDriver(sim, pid_controller)
    >>> 
    >>> # Move to each endpoint and dispense
    >>> for idx, row in df.iterrows():
    ...     if not np.isnan(row['endpoint_robot_x']):
    ...         target = [row['endpoint_robot_x'], row['endpoint_robot_y'], 
    ...                  row['endpoint_robot_z'], 1]
    ...         driver.move_to(target, dispense_pause=60)
"""

import numpy as np
import random
import tempfile
from pathlib import Path
from library.robot_control import *
from library.mask_processing import repair_root_mask_edges
from library.run_inference import run_inference
from library.shoot_mask_cleaning import calculate_global_y_stats, clean_shoot_mask_pipeline
from library.root_shoot_matching import RootMatchingConfig, match_roots_to_shoots_complete


# Global Y-pixel statistics from inference batch

GLOBAL_STATS = {'global_mean': 522.6373561098932,
 'global_std': 40.64363903376843,
 'all_stats': [{'mean': 522.4183627153725, 'std': 67.58658539471621},
  {'mean': 452.95398262798005, 'std': 41.56525140223841},
  {'mean': 490.8661926308985, 'std': 27.346885868291096},
  {'mean': 491.92876883767013, 'std': 24.893401245590812},
  {'mean': 551.0319193616127, 'std': 13.30130229117151},
  {'mean': 543.4997856836691, 'std': 20.34707684591397},
  {'mean': 507.7737608193482, 'std': 58.07985726549244},
  {'mean': 497.31604365278474, 'std': 48.721657949404666},
  {'mean': 482.4776291596063, 'std': 32.85790632020763},
  {'mean': 506.78758850478965, 'std': 20.708410447953284},
  {'mean': 657.317774466855, 'std': 72.20521285868364},
  {'mean': 510.7383455353232, 'std': 30.858226987648244},
  {'mean': 486.4174450352062, 'std': 77.22634688922165},
  {'mean': 536.5708980097133, 'std': 43.147085974995804},
  {'mean': 476.0206638821719, 'std': 21.836197778799253},
  {'mean': 517.2567251461988, 'std': 19.93044627424427},
  {'mean': 638.9460841916085, 'std': 72.86684596583322},
  {'mean': 537.351987845024, 'std': 19.926337738115237},
  {'mean': 522.4358079821378, 'std': 58.824106143078836}],
 'all_means': [522.4183627153725,
  452.95398262798005,
  490.8661926308985,
  491.92876883767013,
  551.0319193616127,
  543.4997856836691,
  507.7737608193482,
  497.31604365278474,
  482.4776291596063,
  506.78758850478965,
  657.317774466855,
  510.7383455353232,
  486.4174450352062,
  536.5708980097133,
  476.0206638821719,
  517.2567251461988,
  638.9460841916085,
  537.351987845024,
  522.4358079821378],
 'all_stds': [67.58658539471621,
  41.56525140223841,
  27.346885868291096,
  24.893401245590812,
  13.30130229117151,
  20.34707684591397,
  58.07985726549244,
  48.721657949404666,
  32.85790632020763,
  20.708410447953284,
  72.20521285868364,
  30.858226987648244,
  77.22634688922165,
  43.147085974995804,
  21.836197778799253,
  19.93044627424427,
  72.86684596583322,
  19.926337738115237,
  58.824106143078836],
 'y_min': 200,
 'y_max': 750}


def pause(sim, steps):
    """Pause simulation for specified number of steps.
    
    Sends zero-velocity commands to maintain robot position without movement.
    
    Args:
        sim (object): Simulation environment instance with run() method.
        steps (int): Number of simulation steps to pause.
        
    Returns:
        None
    """
    sim.run([[0, 0, 0, 0]], num_steps=steps)


class RobotDriver:
    """Driver class for controlling OT-2 robot in simulation.
    
    Provides high-level control interface for moving the robot to target positions
    and dispensing inoculum. Supports both PID and RL-based controllers with
    automatic settling detection and position tolerance checking.
    
    Attributes:
        sim (object): Simulation environment instance.
        controller (object): Either PID controller (with axes attribute) or RL model.
        settle_steps (int): Number of consecutive steps within tolerance to consider settled.
        max_steps (int): Maximum steps allowed per movement attempt.
        pause_steps (int): Default pause duration after dispensing.
        tolerance (float): Position error threshold in meters for settling detection.
        
    Examples:
        >>> # With PID controller
        >>> from library.robot_control import PIDController
        >>> pid = PIDController()
        >>> driver = RobotDriver(sim, pid, settle_steps=10, tolerance=0.0005)
        >>> 
        >>> # Move to position and dispense
        >>> target = [0.15, 0.10, 0.17, 1]  # [x, y, z, dispense]
        >>> success = driver.move_to(target, dispense_pause=60)
        >>> 
        >>> # Move without dispensing
        >>> target = [0.15, 0.10, 0.17, 0]
        >>> success = driver.move_to(target)
    """
    
    def __init__(self, sim, controller, settle_steps=10, max_steps=350, pause_steps=60,
                 tolerance=0.0005):
        """Initialize RobotDriver.
        
        Args:
            sim (object): Simulation environment instance with run() method.
            controller (object): PID controller or RL model for generating actions.
            settle_steps (int, optional): Steps within tolerance to confirm settling. Defaults to 10.
            max_steps (int, optional): Maximum steps per movement. Defaults to 350.
            pause_steps (int, optional): Default pause after dispensing. Defaults to 60.
            tolerance (float, optional): Position error threshold in meters. Defaults to 0.0005.
        """
        self.sim = sim
        self.controller = controller
        self.settle_steps = settle_steps
        self.max_steps = max_steps
        self.pause_steps = pause_steps
        self.tolerance = tolerance
    
    def move_to(self, target, dispense_pause=60, pause_after=False):
        """Move robot to target position with optional dispensing.
        
        Moves the robot end-effector to the specified 3D target position using the
        configured controller. Optionally dispenses inoculum upon arrival and pauses
        afterward for settling.
        
        Args:
            target (list): Target as [x, y, z, dispense] where x, y, z are floats in meters
                          and dispense is 0 (no dispense) or 1 (dispense).
            dispense_pause (int, optional): Pause duration after dispensing in steps. Defaults to 60.
            pause_after (bool, optional): Whether to pause after movement completes. Defaults to False.
            
        Returns:
            bool: True if target was reached within tolerance, False otherwise.
            
        Raises:
            ValueError: If target is not a 4-element list.
            
        Examples:
            >>> # Move and dispense
            >>> success = driver.move_to([0.15, 0.10, 0.17, 1], dispense_pause=60)
            >>> 
            >>> # Move only
            >>> success = driver.move_to([0.15, 0.10, 0.17, 0])
        """
        # Parse target [x, y, z, dispense]
        if len(target) != 4:
            raise ValueError("Target must be [x, y, z, dispense] where dispense is 0 or 1")

        target_pos = {'x': target[0], 'y': target[1], 'z': target[2]}
        dispense = bool(target[3])

        is_pid = self._is_pid_controller()

        if is_pid:
            success = self._move_to_pid(target_pos)
        else:
            success = self._move_to_rl(target_pos)

        if success and dispense:
            self._dispense(dispense_pause=dispense_pause)

        if success and pause_after:
            pause(self.sim, self.pause_steps)

        return success

    def move_to_with_metrics(self, target, dispense_pause=60, pause_after=False):
        """Move robot to target with comprehensive metrics tracking.
        
        High-level method that moves to target, tracks steps taken, calculates
        final position error, and handles dispensing. Automatically detects
        controller type (PID or RL) and uses appropriate control method.
        
        Args:
            target (list): Target as [x, y, z, dispense] where x, y, z are floats 
                in meters and dispense is 0 (no dispense) or 1 (dispense).
            dispense_pause (int, optional): Pause duration after dispensing in steps.
                Defaults to 60.
            pause_after (bool, optional): Whether to pause after movement completes.
                Defaults to False.
                
        Returns:
            dict: Metrics dictionary containing:
                - 'success' (bool): Whether target was reached within tolerance
                - 'steps' (int): Number of steps taken
                - 'target' (dict): Target coordinates {'x', 'y', 'z'}
                - 'final_position' (dict): Final position {'x', 'y', 'z'}
                - 'error_euclidean' (float): 3D Euclidean distance error in meters
                - 'error_x' (float): X-axis error in meters
                - 'error_y' (float): Y-axis error in meters
                - 'error_z' (float): Z-axis error in meters
                - 'dispensed' (bool): Whether dispensing occurred
                
        Raises:
            ValueError: If target is not a 4-element list.
            
        Example:
            >>> target = [0.15, 0.10, 0.17, 1]
            >>> metrics = driver.move_to_with_metrics(target)
            >>> print(f"Success: {metrics['success']}, Steps: {metrics['steps']}")
            >>> print(f"Error: {metrics['error_euclidean']*1000:.3f}mm")
        """
        if len(target) != 4:
            raise ValueError("Target must be [x, y, z, dispense]")
        
        target_dict = {'x': target[0], 'y': target[1], 'z': target[2]}
        dispense = target[3]
        
        # Move to target using appropriate controller
        if self._is_pid_controller():
            success, steps = self._move_to_pid_metrics(target_dict, tolerance=self.tolerance)
        else:
            success, steps = self._move_to_rl_metrics(target_dict, tolerance=self.tolerance)
        
        # Get final position
        final_pos = get_position(self.sim)
        final_dict = {'x': final_pos[0], 'y': final_pos[1], 'z': final_pos[2]}
        
        # Calculate errors
        error_x = target_dict['x'] - final_pos[0]
        error_y = target_dict['y'] - final_pos[1]
        error_z = target_dict['z'] - final_pos[2]
        error_euclidean = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        # Handle dispensing
        dispensed = False
        if dispense == 1 and success:
            self.sim.dispense()
            pause(self.sim, dispense_pause)
            dispensed = True
        
        if pause_after:
            pause(self.sim, self.pause_steps)
        
        # Build metrics dictionary
        metrics = {
            'success': success,
            'steps': steps,
            'target': target_dict.copy(),
            'final_position': final_dict,
            'error_euclidean': float(error_euclidean),
            'error_x': float(error_x),
            'error_y': float(error_y),
            'error_z': float(error_z),
            'dispensed': dispensed
        }
        
        return metrics


    def _dispense(self, dispense_pause=60):
        """Dispense a single drop with variable pause.
        
        Args:
            dispense_pause (int, optional): Pause duration in simulation steps. Defaults to 60.
            
        Returns:
            None
        """
        dispense = [[0, 0, 0, 1]]
        pause = [[0, 0, 0, 0]]
        self.sim.run(dispense, num_steps=1)
        self.sim.run(pause, num_steps=dispense_pause)

    def _is_pid_controller(self):
        """Check if controller is PID or RL model.
        
        Returns:
            bool: True if controller is PID (has axes attribute), False for RL.
        """
        return hasattr(self.controller, 'axes')

    def _move_to_pid(self, target):
        """Move to target using PID controller with settling detection.
        
        Args:
            target (dict): Target position with keys 'x', 'y', 'z' in meters.
            
        Returns:
            bool: True if target reached and settled, False if max_steps exceeded.
        """
        pid_controller = self.controller
        for axis in ['x', 'y', 'z']:
            pid_controller.axes[axis].setpoint = target[axis]

        settled_count = 0 
        
        for step in range(self.max_steps):
            current_pos = get_position(self.sim)
            current_dict = {'x': float(current_pos[0]), 'y': float(current_pos[1]), 'z': float(current_pos[2])}

            error_x = target['x'] - current_pos[0]
            error_y = target['y'] - current_pos[1]
            error_z = target['z'] - current_pos[2]
            total_error = (error_x**2 + error_y**2 + error_z**2)**0.5

            velocities = pid_controller(current_dict, 0.01)
            
            # Check both position error and velocity magnitude
            velocity_magnitude = (velocities['x']**2 + velocities['y']**2 + velocities['z']**2)**0.5
            
            if total_error <= self.tolerance and velocity_magnitude <= 0.01:
                settled_count += 1
                if settled_count >= self.settle_steps:
                    print(f'target reached and settled at step {step}')
                    return True
            else:
                settled_count = 0
        
            actions = [[velocities['x'], velocities['y'], velocities['z'], 0]]
            self.sim.run(actions, 1)
        
        return False
    
    def _move_to_pid_metrics(self, target, tolerance=0.001, velocity_threshold=0.01):
        """
        Move to target using PID controller with settling detection and step tracking.
        
        Args:
            target (dict): Target position with keys 'x', 'y', 'z' in meters.
            tolerance (float, optional): Distance threshold in meters. Defaults to 0.001.
            velocity_threshold (float, optional): Maximum velocity magnitude for settling.
                Defaults to 0.01 m/s.
                
        Returns:
            tuple: (success, steps) where success is bool and steps is int.
        """
        pid_controller = self.controller
        
        # Set setpoint for each axis
        for axis in ['x', 'y', 'z']:
            pid_controller.axes[axis].setpoint = target[axis]

        settled_count = 0 
        steps = 0
        
        for step in range(self.max_steps):
            steps += 1
            
            current_pos = get_position(self.sim)
            current_dict = {'x': float(current_pos[0]), 'y': float(current_pos[1]), 'z': float(current_pos[2])}

            error_x = target['x'] - current_pos[0]
            error_y = target['y'] - current_pos[1]
            error_z = target['z'] - current_pos[2]
            total_error = (error_x**2 + error_y**2 + error_z**2)**0.5

            velocities = pid_controller(current_dict, 0.01)
            
            # Check both position error and velocity magnitude
            velocity_magnitude = (velocities['x']**2 + velocities['y']**2 + velocities['z']**2)**0.5
            
            if total_error <= self.tolerance and velocity_magnitude <= velocity_threshold:
                settled_count += 1
                if settled_count >= self.settle_steps:
                    print(f'target reached and settled at step {step}')
                    return True, steps
            else:
                settled_count = 0
        
            actions = [[velocities['x'], velocities['y'], velocities['z'], 0]]
            self.sim.run(actions, 1)
        
        return False, steps


    def _move_to_rl(self, target, tolerance=0.002, velocity_threshold=0.04):
        """
        Move to target using RL controller with single crossing detection.
        
        Uses single crossing approach: stops commanding on first crossing into
        tolerance with sufficiently low velocity. More appropriate for systems
        that naturally hold position when commands stop.
        
        Args:
            target (dict): Target position with keys 'x', 'y', 'z' in meters.
            tolerance (float, optional): Distance threshold in meters. Defaults to 0.002 (2mm).
            velocity_threshold (float, optional): Maximum velocity magnitude for success.
                Defaults to 0.04 m/s (from testing: 92% success at 2mm).
                
        Returns:
            bool: True if target reached with low velocity, False if max_steps exceeded.
            
        Notes:
            - Based on comprehensive testing showing 92% success at 2mm with 0.04 m/s threshold
            - Uses workspace bounds that must match training configuration
            - Model outputs normalized actions [-1, 1] scaled to velocities [-2, 2] m/s
        """
        # Workspace bounds (must match training exactly)
        WORKSPACE_LOW = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        WORKSPACE_HIGH = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        def normalize_position(position):
            """Normalize position from workspace bounds to [-1, 1]."""
            position = np.asarray(position, dtype=np.float32)
            normalized = 2.0 * (position - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW) - 1.0
            return normalized.astype(np.float32)
        
        def scale_action_to_velocity(action, max_velocity=2.0):
            """Scale normalized action [-1, 1] to velocity commands."""
            action = np.asarray(action, dtype=np.float32)
            velocity = action * max_velocity
            return velocity.astype(np.float32)
        
        # Convert target dict to array
        goal_position = np.array([target['x'], target['y'], target['z']], dtype=np.float32)
        goal_normalized = normalize_position(goal_position)
        
        for step in range(self.max_steps):
            # Get current position
            current_pos = np.array(get_position(self.sim), dtype=np.float32)
            current_normalized = normalize_position(current_pos)
            
            # Build observation [current_x, current_y, current_z, goal_x, goal_y, goal_z]
            obs = np.concatenate([current_normalized, goal_normalized], dtype=np.float32)
            
            # Predict action (use deterministic=False for deployment)
            action, _states = self.controller.predict(obs, deterministic=False)
            
            # Scale to velocity
            velocity = scale_action_to_velocity(action)
            
            # Calculate distance and velocity magnitude
            distance = np.linalg.norm(current_pos - goal_position)
            velocity_magnitude = np.linalg.norm(velocity)
            
            # Single crossing check - stop on first crossing with low velocity
            if distance <= tolerance and velocity_magnitude <= velocity_threshold:
                print(f'target reached at step {step}, distance: {distance*1000:.3f}mm, velocity: {velocity_magnitude:.4f}m/s')
                return True
            
            # Send velocity command to simulation
            actions = [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0]]
            self.sim.run(actions, num_steps=1)
        
        # Max steps reached
        final_pos = np.array(get_position(self.sim), dtype=np.float32)
        final_distance = float(np.linalg.norm(final_pos - goal_position))
        print(f'max steps reached, final distance: {final_distance*1000:.3f}mm')
        return False      

    def _move_to_rl_metrics(self, target, tolerance=0.002, velocity_threshold=0.04):
        """
        Move to target using RL controller with single crossing detection and step tracking.
        
        Uses single crossing approach: stops commanding on first crossing into
        tolerance with sufficiently low velocity. More appropriate for systems
        that naturally hold position when commands stop.
        
        Args:
            target (dict): Target position with keys 'x', 'y', 'z' in meters.
            tolerance (float, optional): Distance threshold in meters. Defaults to 0.002 (2mm).
            velocity_threshold (float, optional): Maximum velocity magnitude for success.
                Defaults to 0.04 m/s (from testing: 92% success at 2mm).
                
        Returns:
            tuple: (success, steps) where success is bool and steps is int.
            
        Notes:
            - Based on comprehensive testing showing 92% success at 2mm with 0.04 m/s threshold
            - Uses workspace bounds that must match training configuration
            - Model outputs normalized actions [-1, 1] scaled to velocities [-2, 2] m/s
        """
        # Workspace bounds (must match training exactly)
        WORKSPACE_LOW = np.array([-0.1871, -0.1706, 0.1700], dtype=np.float32)
        WORKSPACE_HIGH = np.array([0.2532, 0.2197, 0.2897], dtype=np.float32)
        
        def normalize_position(position):
            """Normalize position from workspace bounds to [-1, 1]."""
            position = np.asarray(position, dtype=np.float32)
            normalized = 2.0 * (position - WORKSPACE_LOW) / (WORKSPACE_HIGH - WORKSPACE_LOW) - 1.0
            return normalized.astype(np.float32)
        
        def scale_action_to_velocity(action, max_velocity=2.0):
            """Scale normalized action [-1, 1] to velocity commands."""
            action = np.asarray(action, dtype=np.float32)
            velocity = action * max_velocity
            return velocity.astype(np.float32)
        
        # Convert target dict to array
        goal_position = np.array([target['x'], target['y'], target['z']], dtype=np.float32)
        goal_normalized = normalize_position(goal_position)
        
        steps = 0
        
        for step in range(self.max_steps):
            steps += 1
            
            # Get current position
            current_pos = np.array(get_position(self.sim), dtype=np.float32)
            current_normalized = normalize_position(current_pos)
            
            # Build observation [current_x, current_y, current_z, goal_x, goal_y, goal_z]
            obs = np.concatenate([current_normalized, goal_normalized], dtype=np.float32)
            
            # Predict action (use deterministic=False for deployment)
            action, _states = self.controller.predict(obs, deterministic=False)
            
            # Scale to velocity
            velocity = scale_action_to_velocity(action)
            
            # Calculate distance and velocity magnitude
            distance = np.linalg.norm(current_pos - goal_position)
            velocity_magnitude = np.linalg.norm(velocity)
            
            # Single crossing check - stop on first crossing with low velocity
            if distance <= tolerance and velocity_magnitude <= velocity_threshold:
                print(f'target reached at step {step}, distance: {distance*1000:.3f}mm, velocity: {velocity_magnitude:.4f}m/s')
                return True, steps
            
            # Send velocity command to simulation
            actions = [[float(velocity[0]), float(velocity[1]), float(velocity[2]), 0]]
            self.sim.run(actions, num_steps=1)
        
        # Max steps reached
        final_pos = np.array(get_position(self.sim), dtype=np.float32)
        final_distance = float(np.linalg.norm(final_pos - goal_position))
        print(f'max steps reached, final distance: {final_distance*1000:.3f}mm')
        return False, steps

def get_coord_from_plate(plate_image_path, shoot_model, root_model, temp_path=None):
    """Extract root tip coordinates from specimen image using complete CV pipeline.
    
    Runs the full computer vision workflow: U-Net inference for segmentation,
    mask cleaning and repair, root-shoot matching, and coordinate conversion to
    robot frame. Returns DataFrame with measurements and coordinates for all 5 plants.
    
    Args:
        plate_image_path (str or Path): Path to specimen plate image.
        shoot_model (str or Path): Path to trained shoot segmentation U-Net model (.h5).
        root_model (str or Path): Path to trained root segmentation U-Net model (.h5).
        temp_path (str or Path, optional): Directory for intermediate outputs. 
            Creates temporary directory if None. Defaults to None.
            
    Returns:
        pd.DataFrame: DataFrame with 5 rows (one per plant) containing:
            - plant_order: 1-5 (left to right)
            - Plant ID: Image identifier
            - Length (px): Root length in pixels
            - top_node_x/y: Pixel coordinates of root connection point
            - endpoint_x/y: Pixel coordinates of root tip
            - top_node_robot_x/y/z: Robot coordinates of connection point (meters)
            - endpoint_robot_x/y/z: Robot coordinates of root tip (meters)
            
    Examples:
        >>> df = get_coord_from_plate(
        ...     'specimen_01.png',
        ...     'models/shoot_model.h5',
        ...     'models/root_model.h5'
        ... )
        >>> print(df[['plant_order', 'Length (px)', 'endpoint_robot_x', 'endpoint_robot_y']])
        
    Notes:
        - Uses pre-computed GLOBAL_STATS for shoot mask quality validation
        - Applies edge repair to root masks for improved skeleton quality
        - Returns NaN coordinates for ungerminated seeds (no detected root)
        - All robot coordinates are in meters relative to simulation origin
    """
    config = RootMatchingConfig()
    if not temp_path:
        temp_path = tempfile.mkdtemp()
    else:
        temp_path = Path(temp_path)

    root_model = Path(root_model)
    shoot_model = Path(shoot_model)

    images = [str(plate_image_path)]

    # Run root inference
    mask_type = "root"
    n_processed_root, output_dir_root = run_inference(
        model_path=root_model,
        image_paths=images,
        output_dir=temp_path,
        mask_type=mask_type,
        verbose=False
    )

    # Run shoot inference
    mask_type = "shoot"
    n_processed_shoot, output_dir_shoot = run_inference(
        model_path=shoot_model,
        image_paths=images,
        output_dir=temp_path,
        mask_type=mask_type,
        verbose=False
    )

    # Get inference outputs
    root_inference_path = next(output_dir_root.glob('*.png'), None)
    shoot_inference_path = next(output_dir_shoot.glob('*.png'), None)

    assert root_inference_path, "Root inference failed - no output mask found"
    assert shoot_inference_path, "Shoot inference failed - no output mask found"

    # Clean shoot mask
    results = clean_shoot_mask_pipeline(
        shoot_inference_path, 
        GLOBAL_STATS, 
        quality_check_threshold=2.0
    )
    cleaned_shoot_mask = results['cleaned_mask']
    
    # Repair root mask
    repaired_root_mask = repair_root_mask_edges(root_inference_path)

    # Run complete matching pipeline
    df = match_roots_to_shoots_complete(
        shoot_mask=cleaned_shoot_mask,
        root_mask=repaired_root_mask,
        image_path=plate_image_path,
        config=config,
        return_dataframe=True,
        sample_idx=0,
        verbose=True
    )

    return df