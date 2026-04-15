import pybullet as p
import imageio
from PIL import Image
from pathlib import Path

def search_limit(sim, joint_idx, direction, max_steps=1000):
    """
    Run a joint in the positive or negative direction for a set number of steps 
    reporting when the position if it has stopped within the `max_steps` range.

    Args:
        sim (Simulation): The simulation object
        joint_idx (int): integer value of joint
        direction (int): search in the positive or negative direction
        max_steps(int): Maximum steps to search for each limit
    
    Returns
        tuple (min_limit, max_limit) for the joint
    """
    actions = [[0, 0, 0, 0]]
    actions[0][joint_idx] = direction
    
    for _ in range(max_steps):
        sim.run(actions, num_steps=1)
        states = sim.get_states()
        robotId = list(states.keys())[0]
        vel = states[robotId]['joint_states'][f'joint_{joint_idx}']['velocity']
        
        if abs(vel) < 0.001:
            return get_position(sim)[joint_idx]
    
    return None


def find_joint_limit(sim, joint, max_steps=1000, reset=True):
    """
    Find both limits of a joint by moving positive then negative using velocity values
    
    Args:
        sim (Simulation): The simulation object
        joint (str): Which joint to test: 'x', 'y', or 'z'
    max_steps(int): Maximum steps to search for each limit
    reset(bool): Whether to reset simulation before testing
    stable_steps(int): Number of consecutive steps with unchanged position to consider stable
    
    Returns
        tuple (min_limit, max_limit) for the joint
    """
    if reset:
        sim.reset()
    
    joint_map = {'x': 0, 'y': 1, 'z': 2}
    joint_idx = joint_map[joint.lower()]
    
    
    max_limit = search_limit(sim, joint_idx, 1.0, max_steps)
    min_limit = search_limit(sim, joint_idx, -1.0, max_steps)
    
    return (min_limit, max_limit)

def find_joint_limit_pos(sim, joint, max_steps=1000, reset=True, stable_steps=10):
    """
    Find both limits of a joint by moving positive then negative using position values
    
    Args:
        sim (Simulation): The simulation object
        joint (str): Which joint to test: 'x', 'y', or 'z'
    max_steps(int): Maximum steps to search for each limit
    reset(bool): Whether to reset simulation before testing
    stable_steps(int): Number of consecutive steps with unchanged position to consider stable
    
    Returns
        tuple (min_limit, max_limit) for the joint
    """
    if reset:
        sim.reset()
    
    joint_map = {'x': 0, 'y': 1, 'z': 2}
    joint_idx = joint_map[joint.lower()]
    
    def search_limit(direction):
        actions = [[0, 0, 0, 0]]
        actions[0][joint_idx] = direction
        
        prev_position = None
        stable_count = 0
        
        for _ in range(max_steps):
            sim.run(actions, num_steps=1)
            
            current_position = get_position(sim)[joint_idx]
            
            # Check if position hasn't changed
            if prev_position is not None:
                if abs(current_position - prev_position) < 1e-6:  # Position tolerance
                    stable_count += 1
                    if stable_count >= stable_steps:
                        return current_position
                else:
                    stable_count = 0  # Reset counter if position changed
            
            prev_position = current_position
        
        return None
    
    max_limit = search_limit(1.0)
    min_limit = search_limit(-1.0)
    
    return (min_limit, max_limit)

def get_position(sim, robotId=None):
    """Get the position of the pipet using 
    
    Args:
        sim(simulation object)
        robotId(str): entity id to query, None returns first entity, 'all' returns all as dict
        
    Returns:
        list of positions [x, y, z] OR
        dict of list of positions {robotId: [x, y, z]}"""
    states = sim.get_states()

    if isinstance(robotId, str):
        if robotId == "all":
            return  {i: states[i]['pipette_position'] for i in sorted(states.keys())}
        
        

    if not robotId:
        robotId = list(sorted(states.keys()))[0]
    
    return states.get(robotId, {}).get('pipette_position', [None, None, None])
        
def get_torques(sim, robotId=None):
    """Get motor torques for robot joints.
    
    Args:
        sim: Simulation instance
        robotId (str or None): Robot ID to get torques for. 
            - None: Returns torques for first robot
            - "all": Returns torques for all robots as dict
            - Specific ID: Returns torques for that robot
    
    Returns:
        list or dict: List of torques [x, y, z] for single robot, 
                     or dict mapping robot IDs to torque lists for "all"
    """
    states = sim.get_states()

    if isinstance(robotId, str):
        if robotId == "all":
            data = {}
            for id, val in states.items():
                torques = []
                joints = val['joint_states']
                for j in sorted(joints.keys()):
                    torques.append(joints[j]['motor_torque'])
                data[id] = torques
            return data

    if not robotId:
        robotId = list(sorted(states.keys()))[0]
    
    # Get torques for specific robot
    robot_state = states.get(robotId, {})
    joints = robot_state.get('joint_states', {})
    torques = []
    for j in sorted(joints.keys()):
        torques.append(joints[j]['motor_torque'])
    
    return torques if torques else [None, None, None]


def get_velocities(sim, robotId=None):
    """
    Get the velocity of the pipet
    
    Args:
        sim(simulation object)
        robotId(str): entity id to query, None returns first entity, 'all' returns all as dict
        
    Returns:
        list of positions [x, y, z] OR
        dict of list of velocities {robotId: [x, y, z]}
    """    
    states = sim.get_states()

    if isinstance(robotId, str):
        if robotId == "all":
            return {i: [states[i]['joint_states'][j]['velocity'] 
                       for j in sorted(states[i]['joint_states'].keys())] 
                   for i in sorted(states.keys())}
    
    if not robotId:
        robotId = list(sorted(states.keys()))[0]
    
    robot_state = states.get(robotId, {})
    if 'joint_states' not in robot_state:
        return [None, None, None]
    
    return [robot_state['joint_states'][j]['velocity'] 
            for j in sorted(robot_state['joint_states'].keys())]


def find_workspace(sim, max_steps=1000, reset=True):
    """
    Find the workspace limits and center point.
    
    Parameters:
    -----------
    sim : Simulation
        The simulation object
    max_steps : int
        Maximum steps to search for each limit
    reset : bool
        Whether to reset simulation before testing
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'x_min', 'x_max': X axis limits
        - 'y_min', 'y_max': Y axis limits
        - 'z_min', 'z_max': Z axis limits
        - 'center': (x, y, z) tuple of workspace center
        - 'volume': workspace volume in cubic meters
    """
    # Find limits for each axis
    x_limits = find_joint_limit(sim, 'x', max_steps, reset)
    y_limits = find_joint_limit(sim, 'y', max_steps, False)
    z_limits = find_joint_limit(sim, 'z', max_steps, False)
    
    # Ensure min < max for each axis
    x_min, x_max = min(x_limits), max(x_limits)
    y_min, y_max = min(y_limits), max(y_limits)
    z_min, z_max = min(z_limits), max(z_limits)
    
    # Calculate center
    center = (
        (x_min + x_max) / 2.0,
        (y_min + y_max) / 2.0,
        (z_min + z_max) / 2.0
    )
    
    # Calculate volume
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'z_min': z_min,
        'z_max': z_max,
        'center': center,
        'volume': volume
    }





def record_simulation_gif(sim, action_function, filename, fps=25, 
                          capture_every=1, width=640, height=480):
    """
    Record a GIF of any simulation action function.
    
    This is a general-purpose recorder that can wrap any function that
    controls the simulation. The function will execute normally while
    frames are captured in the background.
    
    Args:
        sim: Simulation object (must have render=True).
        action_function: Callable that performs actions in the simulation.
            This function will be executed and its execution recorded.
        filename (str or Path): Output filepath for the GIF (e.g., 'robot_motion.gif').
        fps (int): Frames per second for the output GIF. Defaults to 25.
        capture_every (int): Capture every Nth frame to reduce file size. 
            E.g., capture_every=2 means capture every other frame. Defaults to 1.
        width (int): Width of captured frames in pixels. Defaults to 640.
        height (int): Height of captured frames in pixels. Defaults to 480.
    
    Returns:
        Any: The return value from action_function (if any).
    
    Example:
        >>> def move_robot():
        >>>     for i in range(100):
        >>>         sim.run([[0.1, 0, 0, 0]], num_steps=1)
        >>> 
        >>> record_simulation_gif(sim, move_robot, 'motion.gif')
    """
    import numpy as np
    
    # Check if simulation has rendering enabled
    if not sim.render:
        print("Error: Simulation must be created with render=True to record GIFs")
        return action_function()
    
    frames = []
    frame_count = 0
    
    # Create a frame capture callback
    def capture_frame():
        nonlocal frame_count
        if frame_count % capture_every == 0:
            try:
                # Get camera image
                img_data = p.getCameraImage(
                    width, height,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
                
                # Check if we got valid data (tuple with 5 elements)
                if isinstance(img_data, tuple) and len(img_data) == 5:
                    width_img, height_img, rgb, depth, seg = img_data
                    
                    # Convert rgb to numpy array if it isn't already
                    if not isinstance(rgb, np.ndarray):
                        rgb = np.array(rgb, dtype=np.uint8)
                    
                    # Reshape if needed (flat array to 2D image)
                    if len(rgb.shape) == 1:
                        rgb = rgb.reshape((height, width, 4))
                    
                    # Convert to PIL Image (remove alpha channel)
                    if rgb.shape[2] == 4:
                        frame = Image.fromarray(rgb[:, :, :3], 'RGB')
                    else:
                        frame = Image.fromarray(rgb, 'RGB')
                    frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to capture frame {frame_count}: {e}")
        frame_count += 1
    
    # Monkey-patch sim.run to capture frames
    original_run = sim.run
    
    def run_with_capture(*args, **kwargs):
        result = original_run(*args, **kwargs)
        capture_frame()
        return result
    
    # Temporarily replace sim.run
    sim.run = run_with_capture
    
    try:
        # Execute the action function
        result = action_function()
    finally:
        # Restore original sim.run
        sim.run = original_run
    
    # Save as GIF
    if frames:
        duration = 1.0 / fps  # Duration per frame in seconds
        imageio.mimsave(filename, frames, duration=duration)
        print(f"GIF saved to {filename} ({len(frames)} frames at {fps} fps)")
    else:
        print("Warning: No frames captured! Make sure your action_function calls sim.run()")
    
    return result