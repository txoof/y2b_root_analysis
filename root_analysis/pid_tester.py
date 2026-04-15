import numpy as np
from datetime import datetime
from pathlib import Path
import json
import uuid
import matplotlib.pyplot as plt
from library.robot_control import get_position, find_workspace, get_velocities, get_torques

from library.pid_controller import PID


def calculate_metrics(position_data, time_data, target):
    """Calculate all PID performance metrics.
    
    Args:
        position_data (list): Position values over time.
        time_data (list): Time step values.
        target (float): Target position.
        
    Returns:
        dict: Dictionary containing rise_time, settling_time, overshoot, steady_state_error.
    """
    if not position_data:
        return {
            'rise_time': None,
            'settling_time': None,
            'overshoot': 0.0,
            'steady_state_error': None
        }
    
    position_array = np.array(position_data)
    start_val = position_data[0]
    
    # Avoid division by zero
    if abs(target - start_val) < 1e-12:
        return {
            'rise_time': 0.0,
            'settling_time': 0.0,
            'overshoot': 0.0,
            'steady_state_error': abs(position_array[-1] - target)
        }
    
    # Calculate rise time (10% to 90%)
    ten_percent = start_val + 0.1 * (target - start_val)
    ninety_percent = start_val + 0.9 * (target - start_val)
    moving_positive = target > start_val
    
    ten_idx = None
    ninety_idx = None
    
    for i, pos in enumerate(position_data):
        if ten_idx is None:
            if (moving_positive and pos >= ten_percent) or (not moving_positive and pos <= ten_percent):
                ten_idx = i
        if ninety_idx is None:
            if (moving_positive and pos >= ninety_percent) or (not moving_positive and pos <= ninety_percent):
                ninety_idx = i
                break
    
    rise_time = time_data[ninety_idx] - time_data[ten_idx] if (ten_idx is not None and ninety_idx is not None) else None
    
    # Calculate settling time (2% band)
    settling_band = abs(target - start_val) * 0.02
    settling_time = None
    
    for i in range(len(position_data) - 1, -1, -1):
        if abs(position_data[i] - target) > settling_band:
            settling_time = time_data[i] if i < len(time_data) - 1 else time_data[-1]
            break
    
    if settling_time is None:
        settling_time = 0
    
    # Calculate overshoot
    if target > start_val:
        max_val = np.max(position_array)
        overshoot = (max_val - target) / abs(target - start_val) * 100 if max_val > target else 0
    else:
        min_val = np.min(position_array)
        overshoot = (target - min_val) / abs(target - start_val) * 100 if min_val < target else 0
    
    # Calculate steady state error
    steady_state_error = abs(position_array[-1] - target)
    
    return {
        'rise_time': rise_time,
        'settling_time': settling_time,
        'overshoot': overshoot,
        'steady_state_error': steady_state_error
    }


class PIDTester:
    """PID controller testing class.
    
    Args:
        sim (Simulation): The simulation object.
        axis (str): Which axis to test ('x', 'y', or 'z'). Defaults to 'x'.
        limits (dict, optional): Workspace limits (auto-calculated if not provided).
        invert_output (bool): Whether to invert the PID output for this axis. Defaults to False.
    """
    
    def __init__(self, sim, axis='x', limits=None, invert_output=False):
        self.sim = sim
        self.axis = axis.lower()
        self.axis_map = {'x': 0, 'y': 1, 'z': 2}
        self.limits = limits
        self.invert_output = invert_output
                
    @property
    def limits(self):
        if self._limits is None:
            w = find_workspace(self.sim)
            self._limits = {
                'x': [w['x_min'], w['x_max']],
                'y': [w['y_min'], w['y_max']],
                'z': [w['z_min'], w['z_max']],
                'center': w['center']
            }
        return self._limits

    @limits.setter
    def limits(self, workspace=None):
        if workspace is not None:
            self._limits = {
                'x': [workspace['x_min'], workspace['x_max']],
                'y': [workspace['y_min'], workspace['y_max']],
                'z': [workspace['z_min'], workspace['z_max']],
                'center': workspace['center']
            }
        else:
            self._limits = None

    def _run_single_trial(self, kp, ki, kd, dt, target_pos,
                          max_steps, tolerance, settle_steps,
                          reset=True, verbose=False):
        """Run a single trial of PID test.

        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            dt (float): Time step.
            target_pos (list): Target position [x, y, z].
            max_steps (int): Maximum number of simulation steps.
            tolerance (float): Position tolerance for settling.
            settle_steps (int): Number of steps within tolerance to consider settled.
            reset (bool): Whether to reset simulation before trial. Defaults to True.
            verbose (bool): Print debug information. Defaults to False.

        Returns:
            dict: Single trial results containing metrics, position data, and metadata.
        """
        if reset:
            self.sim.reset()
        
        axis_idx = self.axis_map[self.axis.lower()]
        target = target_pos[axis_idx]
        start_pos = get_position(self.sim)

        # Check target is within limits
        axis_limits = self.limits[self.axis]
        if target < axis_limits[0] or target > axis_limits[1]:
            raise ValueError(f'Target {target:.4f} outside of {self.axis}-axis range [{axis_limits[0]:.4f}, {axis_limits[1]:.4f}]')

        # Move to starting position
        actions = [[0, 0, 0, 0]]
        actions[0][axis_idx] = -1000
        self.sim.run(actions, 100)

        pid = PID(kp=kp, ki=ki, kd=kd, setpoint=target, invert_output=self.invert_output)

        # Data collection
        position_data = []
        time_data = []
        error_data = []
        control_data = []
        velocity_data = []
        torque_data = []
        
        # Settling detection
        steps_in_tolerance = 0

        for step in range(max_steps):
            current_pos = get_position(self.sim)
            current_val = current_pos[axis_idx]
            current_vel = get_velocities(self.sim)
            current_torque = get_torques(self.sim)
            
            if verbose:
                print(f'current_val {self.axis} axis: {current_val:.5f} -> {target:.5f}')
            
            position_data.append(current_val)
            time_data.append(step)
            error_data.append(target - current_val)
            velocity_data.append(current_vel[axis_idx])
            torque_data.append(current_torque[axis_idx])
            
            # Check position remains near target
            if abs(current_val - target) <= tolerance:
                steps_in_tolerance += 1
                if verbose:
                    print(f'settling {steps_in_tolerance} of {settle_steps}')
            else:
                steps_in_tolerance = 0

            # Stop execution if position remains near target
            if steps_in_tolerance >= settle_steps:
                break

            # Calculate PID output 
            vx = pid(current_val, dt)
            control_data.append(vx)

            actions = [[0, 0, 0, 0]]
            actions[0][axis_idx] = vx
            self.sim.run(actions, num_steps=1)

        # Calculate metrics
        metrics = calculate_metrics(position_data, time_data, target)
        
        # Store result
        result = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'position_data': position_data,
            'time_data': time_data,
            'error_data': error_data,
            'control_data': control_data,
            'velocity_data': velocity_data,
            'torque_data': torque_data,
            'target': target,
            'start_position': start_pos[self.axis_map[self.axis.lower()]],
            'settled': steps_in_tolerance >= settle_steps,
            'invert_output': self.invert_output,
            'total_time': len(time_data),

        }
        
        return result
    
    def test_gains(self, kp, ki, kd, dt, target_pos,
                   max_steps, tolerance, settle_steps, num_trials,
                   reset=True, verbose=False):
        """Test a set of PID gains over multiple trials.
        
        Args:
            kp (float): Proportional gain.
            ki (float): Integral gain.
            kd (float): Derivative gain.
            dt (float): Time step.
            target_pos (list): Target position [x, y, z].
            max_steps (int): Maximum number of simulation steps.
            tolerance (float): Position tolerance for settling.
            settle_steps (int): Number of steps within tolerance to consider settled.
            num_trials (int): Number of trials to run.
            reset (bool): Whether to reset simulation before each trial. Defaults to True.
            verbose (bool): Print debug information. Defaults to False.
        
        Returns:
            dict: Test results with gains, timestamp, and trial data.
        """
        trials = []

        for trial in range(num_trials):
            trial_result = self._run_single_trial(
                kp, ki, kd, dt, target_pos,
                max_steps, tolerance, settle_steps,
                reset=reset, verbose=verbose
            )
            trials.append(trial_result)

        result = {
            'gains': {'kp': kp, 'ki': ki, 'kd': kd},
            'timestamp': datetime.now().isoformat(),
            'num_trials': num_trials,
            'trials': trials
        }

        return result
    

class Experiment:
    """Create a new experiment to track PID testing.
    
    Args:
        title (str): Title of the experiment.
        hypothesis (str): Hypothesis being tested.
        axis (str): Axis being tested ('x', 'y', or 'z').
        invert_output (bool): Whether output was inverted for this axis. Defaults to False.
    """
    
    def __init__(self, title, hypothesis, axis, invert_output=False):
        self.id = str(uuid.uuid4())
        self.title = title
        self.hypothesis = hypothesis
        self.axis = axis.lower()
        self.timestamp = datetime.now().isoformat()
        self.tests = []
        self.invert_output = invert_output
    
    def add_test(self, test_result):
        """Add a test result from PIDTester.test_gains().
        
        Args:
            test_result (dict): Result dictionary from test_gains().
        
        Raises:
            ValueError: If test_result doesn't contain required keys.
        """
        # Basic validation
        if 'gains' not in test_result or 'trials' not in test_result:
            raise ValueError("Invalid test_result: must contain 'gains' and 'trials'")
        
        self.tests.append(test_result)
    
    def get_averaged_metrics(self, test_index):
        """Get averaged metrics across trials for a specific test.
        
        Args:
            test_index (int): Index of the test in self.tests.
        
        Returns:
            dict: Averaged metrics with mean, std, min, max for each metric.
        
        Raises:
            IndexError: If test_index is out of range.
        """
        if test_index >= len(self.tests):
            raise IndexError(f"Test index {test_index} out of range")
        
        test = self.tests[test_index]
        trials = test['trials']
        
        if len(trials) == 1:
            return trials[0]['metrics']
        
        metric_keys = ['rise_time', 'settling_time', 'overshoot', 'steady_state_error']
        averaged = {}
        
        for key in metric_keys:
            values = [t['metrics'][key] for t in trials if t['metrics'][key] is not None]
            if values:
                averaged[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
            else:
                averaged[key] = None
        
        return averaged
    
    def print_summary(self):
        """Print summary table of all tests in this experiment."""
        print(f"\nExperiment: {self.title}")
        print(f"Hypothesis: {self.hypothesis}")
        print(f"Axis: {self.axis}")
        print(f"Inverted Output: {self.invert_output}")
        print(f"Tests: {len(self.tests)}")
        print("\n" + "="*90)  # <-- MAKE WIDER
        print(f"{'Kp':>8} {'Ki':>8} {'Kd':>8} {'Rise':>10} {'Settling':>10} {'Overshoot':>10} {'SS Error':>10} {'Total Steps':>8}")  # <-- ADD COLUMN
        print("="*90)
        
        for i, test in enumerate(self.tests):
            gains = test['gains']
            metrics = self.get_averaged_metrics(i)
            
            # Handle both single trial (dict) and averaged (dict with 'mean')
            if isinstance(metrics['rise_time'], dict):
                rise = metrics['rise_time']['mean']
                settling = metrics['settling_time']['mean']
                overshoot = metrics['overshoot']['mean']
                ss_error = metrics['steady_state_error']['mean']
            else:
                rise = metrics['rise_time']
                settling = metrics['settling_time']
                overshoot = metrics['overshoot']
                ss_error = metrics['steady_state_error']
            
            # Get total time from first trial
            total_time = test['trials'][0]['total_time']
            
            # Format None values as 'N/A' instead of trying to format as float
            rise_str = f"{rise:>10.2f}" if rise is not None else f"{'N/A':>10}"
            settling_str = f"{settling:>10.2f}" if settling is not None else f"{'N/A':>10}"
            overshoot_str = f"{overshoot:>10.2f}" if overshoot is not None else f"{'N/A':>10}"
            ss_error_str = f"{ss_error:>10.6f}" if ss_error is not None else f"{'N/A':>10}"
            
            print(f"{gains['kp']:>8.2f} {gains['ki']:>8.2f} {gains['kd']:>8.2f} "
                f"{rise_str} {settling_str} {overshoot_str} {ss_error_str} {total_time:>8}")  # <-- ADD TOTAL
        
        print("="*90)
    
    def _determine_varying_gain(self):
        """Determine which gain parameter varies across tests.
        
        Returns:
            tuple: (gain_key, x_values, x_labels) where:
                - gain_key (str or None): 'Kp', 'Ki', 'Kd', or None if multiple vary.
                - x_values (list): Numeric values for x-axis.
                - x_labels (list): String labels for x-axis.
        """
        gains_list = [test['gains'] for test in self.tests]
        
        kp_values = [g['kp'] for g in gains_list]
        ki_values = [g['ki'] for g in gains_list]
        kd_values = [g['kd'] for g in gains_list]
        
        kp_varies = len(set(kp_values)) > 1
        ki_varies = len(set(ki_values)) > 1
        kd_varies = len(set(kd_values)) > 1
        
        varies_count = sum([kp_varies, ki_varies, kd_varies])
        
        if varies_count == 0:
            # All tests have same gains
            return None, list(range(len(self.tests))), [str(i) for i in range(len(self.tests))]
        elif varies_count == 1:
            # Exactly one gain varies
            if kp_varies:
                # Format labels to avoid overlapping decimals
                labels = [f"{v:.0f}" if v >= 1 else f"{v:.2f}" for v in kp_values]
                return 'Kp', kp_values, labels
            elif ki_varies:
                labels = [f"{v:.0f}" if v >= 1 else f"{v:.2f}" for v in ki_values]
                return 'Ki', ki_values, labels
            else:
                labels = [f"{v:.0f}" if v >= 1 else f"{v:.2f}" for v in kd_values]
                return 'Kd', kd_values, labels
        else:
            # Multiple gains vary - use test index with composite labels
            x_values = list(range(len(self.tests)))
            x_labels = [f"P={g['kp']:.0f if g['kp']>=1 else g['kp']:.2f},I={g['ki']:.0f if g['ki']>=1 else g['ki']:.2f},D={g['kd']:.0f if g['kd']>=1 else g['kd']:.2f}" for g in gains_list]
            return None, x_values, x_labels

    def plot_metrics_summary(self, metric_names='all', separate_subplots=True, plot_type='lines'):
        """Plot averaged metrics across all tests.
        
        Args:
            metric_names (list of str or 'all', optional): Which metrics to plot for line plots.
                If 'all' or None, plots all metrics. Ignored for scatter plots.
                Options: 'rise_time', 'settling_time', 'overshoot', 'steady_state_error'.
            separate_subplots (bool): If True, each metric gets its own subplot. 
                If False, all on one plot. Only applies to line plots. Defaults to True.
            plot_type (str): Type of plot - 'lines' for metric trends or 'scatter' for 
                speed vs stability tradeoff. Defaults to 'lines'.
        
        Returns:
            tuple: (fig, axes) Matplotlib figure and axes objects, or (None, None) if no tests.
        """
        if not self.tests:
            print("No tests to plot")
            return None, None
        
        # Scatter plot: speed vs stability
        if plot_type == 'scatter':
            return self._plot_speed_vs_stability()
        
        # Line plot: existing behavior
        if metric_names == 'all' or metric_names is None:
            metric_names = ['rise_time', 'settling_time', 'overshoot', 'steady_state_error']
        
        # Determine which gain varies
        gain_key, x_values, x_labels = self._determine_varying_gain()
        
        # Collect data
        data = {metric: [] for metric in metric_names}
        errors = {metric: [] for metric in metric_names}
        valid_indices = {metric: [] for metric in metric_names}
        
        for i, test in enumerate(self.tests):
            metrics = self.get_averaged_metrics(i)
            for metric in metric_names:
                if isinstance(metrics[metric], dict):
                    value = metrics[metric]['mean']
                    error = metrics[metric]['std']
                else:
                    value = metrics[metric]
                    error = 0
                
                if value is not None:
                    data[metric].append(value)
                    errors[metric].append(error)
                    valid_indices[metric].append(i)
        
        # Create plots
        if separate_subplots:
            fig, axes = plt.subplots(len(metric_names), 1, figsize=(10, 4*len(metric_names)))
            if len(metric_names) == 1:
                axes = [axes]
            
            for ax, metric in zip(axes, metric_names):
                if data[metric]:
                    valid_x_values = [x_values[i] for i in valid_indices[metric]]
                    valid_x_labels = [x_labels[i] for i in valid_indices[metric]]
                    
                    ax.errorbar(valid_x_values, data[metric], yerr=errors[metric], 
                            marker='o', capsize=5, capthick=2)
                    ax.set_xlabel(gain_key if gain_key else 'Test Index')
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_xscale('log')
                    ax.set_xticks(valid_x_values)
                    ax.set_xticklabels(valid_x_labels)
                    ax.grid(True, alpha=0.3, which='both')
                else:
                    ax.text(0.5, 0.5, f'No valid data for {metric}', 
                        ha='center', va='center', transform=ax.transAxes)
                    ax.set_xlabel(gain_key if gain_key else 'Test Index')
                    ax.set_ylabel(metric.replace('_', ' ').title())
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            for metric in metric_names:
                if data[metric]:
                    valid_x_values = [x_values[i] for i in valid_indices[metric]]
                    
                    ax.errorbar(valid_x_values, data[metric], yerr=errors[metric],
                            marker='o', capsize=5, capthick=2, label=metric.replace('_', ' ').title())
            
            ax.set_xlabel(gain_key if gain_key else 'Test Index')
            ax.set_ylabel('Value')
            ax.set_xscale('log')
            if x_values:
                ax.set_xticks(x_values)
                ax.set_xticklabels(x_labels)
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
        
        fig.suptitle(f"{self.title}\n{self.hypothesis}", fontsize=12)
        fig.tight_layout()
        
        return fig, axes if separate_subplots else ax

    def _plot_speed_vs_stability(self):
        """Create scatter plot of speed (total time) vs stability (overshoot).
        
        Returns:
            tuple: (fig, ax) Matplotlib figure and axis objects.
        """
        # Collect data
        total_times = []
        overshoots = []
        gain_values = []
        
        for i, test in enumerate(self.tests):
            metrics = self.get_averaged_metrics(i)
            
            # Get overshoot
            if isinstance(metrics['overshoot'], dict):
                overshoot = metrics['overshoot']['mean']
            else:
                overshoot = metrics['overshoot']
            
            # Get total time from first trial
            total_time = test['trials'][0]['total_time']
            
            if overshoot is not None and total_time is not None:
                total_times.append(total_time)
                overshoots.append(overshoot)
                gain_values.append(i)
        
        if not total_times:
            print("No valid data for scatter plot")
            return None, None
        
        # Determine which gain varies for coloring
        gain_key, varying_gains, _ = self._determine_varying_gain()
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Use colormap based on varying gain values
        if gain_key:
            # Extract the actual varying gain values for valid indices
            color_values = [varying_gains[i] for i in gain_values]
            scatter = ax.scatter(total_times, overshoots, c=color_values, 
                            cmap='viridis', s=100, edgecolors='black', linewidth=1.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(gain_key, rotation=270, labelpad=20)
        else:
            # No single varying gain, use uniform color
            ax.scatter(total_times, overshoots, s=100, edgecolors='black', linewidth=1.5)
        
        # Annotate points with gain values
        gains_list = [self.tests[i]['gains'] for i in gain_values]
        for i, (x, y, gains) in enumerate(zip(total_times, overshoots, gains_list)):
            # Format gain values outside f-string
            kp_str = f"{gains['kp']:.0f}" if gains['kp'] >= 1 else f"{gains['kp']:.2f}"
            ki_str = f"{gains['ki']:.0f}" if gains['ki'] >= 1 else f"{gains['ki']:.2f}"
            kd_str = f"{gains['kd']:.0f}" if gains['kd'] >= 1 else f"{gains['kd']:.2f}"
            
            label = f"P={kp_str}\nI={ki_str}\nD={kd_str}"
            ax.annotate(label, (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Total Time (steps)', fontsize=12)
        ax.set_ylabel('Overshoot (%)', fontsize=12)
        ax.set_title(f"{self.title}\nSpeed vs Stability Tradeoff", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add arrow pointing to ideal region (bottom-left)
        ax.annotate('Ideal\n(Fast & Stable)', xy=(min(total_times), min(overshoots)),
                xytext=(0.05, 0.95), textcoords='axes fraction',
                fontsize=10, color='green', weight='bold',
                ha='left', va='top',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        fig.tight_layout()
        
        return fig, ax
    
    def plot_trajectory(self, test_index, trial_index=0):
        """Plot detailed trajectory data for a specific trial.
        
        Shows position, error, and control signal over time.
        
        Args:
            test_index (int): Index of the test in self.tests.
            trial_index (int): Index of the trial within that test. Defaults to 0.
        
        Returns:
            tuple: (fig, axes) Matplotlib figure and axes objects (3 subplots).
        
        Raises:
            IndexError: If test_index or trial_index is out of range.
        """
        if test_index >= len(self.tests):
            raise IndexError(f"Test index {test_index} out of range")
        
        test = self.tests[test_index]
        
        if trial_index >= len(test['trials']):
            raise IndexError(f"Trial index {trial_index} out of range")
        
        trial = test['trials'][trial_index]
        gains = test['gains']
        
        # Extract data
        time_data = trial['time_data']
        velocity_data = trial['velocity_data']
        torque_data = trial['torque_data']
        position_data = trial['position_data']
        error_data = trial['error_data']
        control_data = trial['control_data']
        target = trial['target']
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        
        # Plot 1: Position vs Time
        axes[0].plot(time_data, position_data, 'b-', linewidth=2, label='Actual Position')
        axes[0].axhline(y=target, color='r', linestyle='--', linewidth=1, label='Target')
        axes[0].set_xlabel('Time (steps)')
        axes[0].set_ylabel('Position')
        axes[0].set_title('Position vs Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Error vs Time
        axes[1].plot(time_data, error_data, 'g-', linewidth=2)
        axes[1].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Time (steps)')
        axes[1].set_ylabel('Error')
        axes[1].set_title('Error vs Time')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Control Signal vs Time
        axes[2].plot(time_data[:len(control_data)], control_data, 'orange', linewidth=2)
        axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[2].set_xlabel('Time (steps)')
        axes[2].set_ylabel('Control Signal')
        axes[2].set_title('Control Signal vs Time')
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Velocity vs Time
        axes[3].plot(time_data, velocity_data, 'purple', linewidth=2)
        axes[3].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[3].set_xlabel('Time (steps)')
        axes[3].set_ylabel('Velocity')
        axes[3].set_title('Velocity vs Time')
        axes[3].grid(True, alpha=0.3)
        
        # Plot 5: Torque vs Time
        axes[4].plot(time_data[:len(torque_data)], torque_data, 'brown', linewidth=2)
        axes[4].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[4].set_xlabel('Time (steps)')
        axes[4].set_ylabel('Torque')
        axes[4].set_title('Torque vs Time')
        axes[4].grid(True, alpha=0.3)
        # Overall title
        fig.suptitle(
            f"{self.title}\nKp={gains['kp']}, Ki={gains['ki']}, Kd={gains['kd']} | Trial {trial_index}",
            fontsize=12
        )
        fig.tight_layout()
        
        return fig, axes
    
    def to_dict(self):
        """Convert experiment to dictionary for JSON serialization.
        
        Returns:
            dict: Dictionary representation of the experiment.
        """
        return {
            'id': self.id,
            'title': self.title,
            'hypothesis': self.hypothesis,
            'axis': self.axis,
            'timestamp': self.timestamp,
            'tests': self.tests,
            'invert_output': self.invert_output
        }


class ExperimentLog:
    """Manage a log of PID experiments.
    
    Args:
        filepath (str or Path): Path to the JSON log file.
    """
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.experiments = []
    
    def add_experiment(self, experiment):
        """Add an Experiment to the log.
        
        Args:
            experiment (Experiment): Experiment object to add.
        
        Raises:
            TypeError: If experiment is not an Experiment object.
        """
        if not isinstance(experiment, Experiment):
            raise TypeError("Must add an Experiment object")
        self.experiments.append(experiment)
    
    def save(self):
        """Save experiments to JSON file, preserving existing experiments.
        
        Checks for duplicate IDs and updates existing experiments.
        """
        # Load existing data if file exists
        if self.filepath.exists():
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            existing_experiments = data.get('experiments', [])
        else:
            existing_experiments = []
        
        # Create dict of existing experiments by ID
        existing_by_id = {exp['id']: exp for exp in existing_experiments}
        
        # Add or update with current experiments
        for exp in self.experiments:
            existing_by_id[exp.id] = exp.to_dict()
        
        # Save all experiments
        data = {'experiments': list(existing_by_id.values())}
        
        with open(self.filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.experiments)} experiments to {self.filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load experiments from JSON file.
        
        Args:
            filepath (str or Path): Path to the JSON log file.
        
        Returns:
            ExperimentLog: Log with loaded Experiment objects.
        """
        log = cls(filepath)
        
        if not log.filepath.exists():
            print(f"No existing file at {filepath}, starting new log")
            return log
        
        with open(log.filepath, 'r') as f:
            data = json.load(f)
        
        for exp_dict in data.get('experiments', []):
            exp = cls._from_dict(exp_dict)
            log.experiments.append(exp)
        
        print(f"Loaded {len(log.experiments)} experiments from {filepath}")
        return log
    
    @staticmethod
    def _from_dict(exp_dict):
        """Reconstruct an Experiment object from a dictionary.
        
        Args:
            exp_dict (dict): Dictionary representation of an experiment.
        
        Returns:
            Experiment: Reconstructed Experiment object.
        """
        exp = Experiment(
            title=exp_dict['title'],
            hypothesis=exp_dict['hypothesis'],
            axis=exp_dict['axis'],
            invert_output=exp_dict.get('invert_output', False)
        )
        exp.id = exp_dict['id']
        exp.timestamp = exp_dict['timestamp']
        exp.tests = exp_dict['tests']
        return exp
    
    def get_experiment(self, identifier):
        """Get experiment by title or short hash (first 6 characters of ID).
        
        Args:
            identifier (str): Either the full/partial title or first 6 chars of ID.
        
        Returns:
            Experiment or None: Matching experiment, or None if not found.
        """
        # Try short hash first
        if len(identifier) == 6:
            for exp in self.experiments:
                if exp.id[:6] == identifier:
                    return exp
        
        # Try title (case-insensitive partial match)
        identifier_lower = identifier.lower()
        for exp in self.experiments:
            if identifier_lower in exp.title.lower():
                return exp
        
        return None
    
    def list_experiments(self):
        """Print summary table of all experiments."""
        if not self.experiments:
            print("No experiments in log")
            return
        
        print("\n" + "="*100)
        print(f"{'ID':>8} {'Title':<30} {'Hypothesis':<40} {'Tests':>6}")
        print("="*100)
        
        for exp in self.experiments:
            short_id = exp.id[:6]
            title = exp.title[:28] + '..' if len(exp.title) > 30 else exp.title
            hypothesis = exp.hypothesis[:38] + '..' if len(exp.hypothesis) > 40 else exp.hypothesis
            num_tests = len(exp.tests)
            
            print(f"{short_id:>8} {title:<30} {hypothesis:<40} {num_tests:>6}")
        
        print("="*100)