class PID:
    """PID controller.

    Implements a simple proportional–integral–derivative controller.

    Args:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        setpoint (float, optional): Desired target value. Defaults to 0.0.
        output_limits (tuple, optional): Tuple (min, max) to clamp the controller output.
            Use None for no limit. Defaults to (None, None).
        invert_output (bool): swap sign on output 

    Attributes:
        kp (float): Proportional gain.
        ki (float): Integral gain.
        kd (float): Derivative gain.
        setpoint (float): Target setpoint.
        min_output (float or None): Minimum output limit.
        max_output (float or None): Maximum output limit.
        invert_output (bool): swap sign on output

    """

    def __init__(self, kp, ki, kd, setpoint=0.0, 
                 output_limits=(None, None),
                 invert_output=False):
        """
        Initialize the PID controller.

        Parameters are the same as described in the class docstring.
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.setpoint = setpoint

        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None

        # Output limits: (min, max)
        self.min_output, self.max_output = output_limits
        self.invert_output = invert_output

    def reset(self):
        """
        Reset the controller internal state.

        Clears the integral accumulator and last error/time so the controller
        behaves as if newly constructed.
        """
        self._last_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def __call__(self, measurement, dt):
        """
        Calculate the PID output for a given measurement and timestep.

        Args:
            measurement (float): The current measured value.
            dt (float): Time interval in seconds since the last call. If dt <= 0.0,
            the derivative term is treated as zero and the integral is not updated.

        Returns:
            float: Control output after applying proportional, integral, and
            derivative terms, clamped to output_limits if specified.
        """
        error = self.setpoint - measurement

        # proportional
        p = self.kp * error

        # integral (sum)
        if dt > 0.0:
            self._integral += error * dt
        i = self.ki * self._integral

        # derivative (slope)
        if dt > 0.0:
            derivative = (error - self._last_error) / dt
        else:
            derivative = 0.0
        d = self.kd * derivative

        output = p + i + d

        # check limits
        if self.min_output is not None:
            output = max(self.min_output, output)
        if self.max_output is not None:
            output = min(self.max_output, output)

        self._last_error = error

        if self.invert_output:
            output = output * -1

        return output


class Controller():
    """Multi-axis PID controller for coordinated control.
    
    Manages multiple PID controllers, one per axis, for coordinated multi-axis
    control systems such as robotics applications. Each axis can have independent
    PID gains and setpoints.
    
    Args:
        axes (dict, optional): Dictionary mapping axis names to PID parameter
            dictionaries. Each parameter dictionary should contain 'kp', 'ki',
            'kd', and 'setpoint' keys. If None, creates default x, y, z axes
            with zero gains. Defaults to None.
    
    Attributes:
        DEFAULT_PID (dict): Default PID parameters with all gains set to zero.
            Used when no axes configuration is provided.
        axes (dict): Dictionary mapping axis names to PID controller instances.
    
    Example:
        >>> axes = {
        ...     'x': {'kp': 37, 'ki': 0, 'kd': 1.2, 'setpoint': 0.2},
        ...     'y': {'kp': 57, 'ki': 0, 'kd': 0.2, 'setpoint': 0.3},
        ...     'z': {'kp': 12, 'ki': 0, 'kd': 2.2, 'setpoint': 0.01},
        ... }
        >>> controller = Controller(axes=axes)
        >>> measurements = {'x': 0.15, 'y': 0.20, 'z': 0.005}
        >>> outputs = controller(measurements, dt=0.01)
    """
    
    DEFAULT_PID = {
        'kp': 0,
        'ki': 0,
        'kd': 0,
        'setpoint': 0
    }
    
    def __init__(self, axes=None):
        """Initialize the multi-axis controller.
        
        Args:
            axes (dict, optional): Dictionary mapping axis names to PID parameter
                dictionaries. Each parameter dictionary should contain 'kp', 'ki',
                'kd', and 'setpoint' keys. If None, creates default x, y, z axes
                with zero gains. Defaults to None.
        
        Example:
            >>> # Custom axes configuration
            >>> axes = {
            ...     'x': {'kp': 10, 'ki': 0.1, 'kd': 1, 'setpoint': 0.5},
            ...     'y': {'kp': 15, 'ki': 0.2, 'kd': 2, 'setpoint': 0.3}
            ... }
            >>> controller = Controller(axes=axes)
            
            >>> # Default configuration (all zeros)
            >>> controller = Controller()
        """
        self.axes = axes
    
    @property
    def axes(self):
        """Get dictionary mapping axis names to PID controller instances.
        
        Returns:
            dict: Dictionary with axis names as keys (e.g., 'x', 'y', 'z') and
                PID controller instances as values.
        """
        return self._axes

    @axes.setter
    def axes(self, axes):
        """Set up PID controllers for each axis.
        
        Creates individual PID controller instances for each axis based on the
        provided configuration. If no configuration is provided, creates default
        x, y, z axes with zero gains.
        
        Args:
            axes (dict or None): Dictionary mapping axis names to PID parameter
                dictionaries, or None for default configuration.
        """
        pid_axes = {}
        if not axes:
            my_axes = {'x': self.DEFAULT_PID,
                       'y': self.DEFAULT_PID,
                       'z': self.DEFAULT_PID}
        else:
            my_axes = axes
        
        for a, value in my_axes.items():
            pid_axes[a] = PID(**value)

        self._axes = pid_axes

    def reset(self, axis='all'):
        """Reset internal state of one or more PID controllers.
        
        Clears the integral accumulator and last error/time for the specified
        axis or axes, returning them to their initial state.
        
        Args:
            axis (str or list, optional): Specifies which axes to reset:
                - 'all': Reset all axes (default)
                - str: Reset single axis by name (e.g., 'x')
                - list: Reset multiple specific axes (e.g., ['x', 'z'])
        
        Raises:
            KeyError: If a specified axis name does not exist.
        
        Example:
            >>> controller.reset()  # Reset all axes
            >>> controller.reset('x')  # Reset only x-axis
            >>> controller.reset(['x', 'y'])  # Reset x and y axes
        """
        if axis == 'all':
            axis_list = list(self.axes.values())
        elif isinstance(axis, str):
            # Single axis name
            axis_list = [self.axes[axis]]
        else:
            # Assume it's an iterable of axis names
            axis_list = [self.axes[a] for a in axis]
        
        for a in axis_list:
            a.reset()

    def __call__(self, measurements, dt):
        """Compute PID output for all axes.
        
        Calculates control outputs for all configured axes based on current
        measurements and the time step. This is the primary method for getting
        control commands during simulation or real-time control.
        
        Args:
            measurements (dict): Current measured values for each axis.
                Keys must match the axis names defined in the controller.
                Example: {'x': 0.15, 'y': 0.20, 'z': 1.05}
            dt (float): Time step in seconds since the last update.
                Must be positive for proper integral and derivative calculations.
        
        Returns:
            dict: Control output (typically velocity commands) for each axis.
                Keys match the input measurement keys.
                Example: {'x': 0.05, 'y': -0.03, 'z': 0.01}
        
        Raises:
            KeyError: If measurements dict is missing a required axis.
        
        Example:
            >>> measurements = {'x': 0.15, 'y': 0.20, 'z': 0.005}
            >>> outputs = controller(measurements, dt=0.01)
            >>> print(outputs)
            {'x': 1.85, 'y': 5.7, 'z': 0.06}
        """
        return {axis: controller(measurements[axis], dt) 
                for axis, controller in self.axes.items()}