# PyBullet OT-2 Simulation - Setup Guide

## Overview

This repository contains a PyBullet simulation for the OT-2 robot. The simulation allows you to test pipetting actions and visualize robot movements.

## Installation

### Standard Installation

```bash
pip install pybullet
```

### macOS Sequoia Installation

PyBullet fails to compile on macOS Sequoia with the default compiler. Use LLVM 16:

```bash
brew install llvm@16
export PATH="/opt/homebrew/opt/llvm@16/bin:$PATH"
export CC="/opt/homebrew/opt/llvm@16/bin/clang"
export CXX="/opt/homebrew/opt/llvm@16/bin/clang++"
pip install pybullet
```

See `pybullet-macos-sequoia-fix.md` for detailed installation instructions.

## Usage

### Basic Example

See `demo.ipynb` for working examples.

```python
from sim_class_FINAL import Simulation
import time

# Create simulation
sim = Simulation(num_agents=1)

# Run actions
actions = [[0.1, 0.1, 0.9, 1]]
sim.run(actions, num_steps=100)

# Wait
time.sleep(2)

# More actions
actions = [[-0.1, -0.2, -0.9, 0]]
sim.run(actions, num_steps=100)

# Reset for next experiment
sim.reset(num_agents=1)

# Run again
actions = [[0.2, 0.2, 0.2, 0]]
sim.run(actions, num_steps=100)
```

### Development Mode (No GUI)

For faster testing without visualization:

```python
from sim_class_FINAL import Simulation

# No GUI = 10x faster
sim = Simulation(num_agents=1, render=False)

# Run your experiments
actions = [[0.1, 0.1, 0.1, 0]]
sim.run(actions, num_steps=100)

# Close when done
sim.close()
```

### Visualization Mode

When you need to see the simulation:

```python
from sim_class_FINAL import Simulation

# Create once
sim = Simulation(num_agents=1, render=True)

# Run experiment 1
actions = [[0.1, 0.1, 0.9, 1]]
sim.run(actions, num_steps=100)

# Reset and run experiment 2
sim.reset(num_agents=1)
actions = [[-0.1, -0.2, -0.9, 0]]
sim.run(actions, num_steps=100)
```

## Best Practices

1. **Use `render=False` for development** - much faster, enables programmatic window closing
2. **Create simulation once, reuse with `reset()`** - more efficient than creating multiple instances
3. **Run multiple experiments in sequence** - reset between experiments rather than creating new simulations

## macOS-Specific Issues

PyBullet has known limitations on macOS due to its single-threaded GUI architecture:

### Known Issues

1. **Spinning wheel on startup** - Window appears frozen until simulation runs
2. **Window freezes between simulations** - GUI only responds during `stepSimulation()` calls
3. **Cannot close windows programmatically** - `disconnect()` cannot be processed by frozen GUI
4. **Window accumulation** - Multiple `Simulation()` objects leave ghost windows

### How `sim_class_FINAL.py` Helps

**Fixed issues:**
- Spinning wheel on startup (auto-calls `stepSimulation()` after initialization)
- Window accumulation (auto-disconnects before creating new connection)
- Reset creates new windows (reuses existing connection)

**Cannot be fixed:**
- GUI windows won't close programmatically (PyBullet architecture limitation)
- Window freezes when simulation stops (inherent to single-threaded GUI)

### Understanding Window Behavior on macOS

**The window is responsive ONLY while simulation is running:**

- **During** `sim.run(actions, num_steps=100)` - Window is interactive (you can rotate, zoom, pan)
- **After** `sim.run()` completes - Window freezes immediately
- **During next** `sim.run()` - Window becomes interactive again

This happens because the GUI event loop only processes when `stepSimulation()` is actively being called.

### macOS Workflow

**In Jupyter notebooks:**

```python
from sim_class_FINAL import Simulation

# Create once at the start
sim = Simulation(num_agents=1, render=True)

# Run all your experiments
for experiment in experiments:
    actions = experiment['actions']
    sim.run(actions, num_steps=100)
    sim.reset(num_agents=1)

# Don't call close() - restart kernel when completely done
# Kernel > Restart Kernel
```

**Why not call close()?**

The window is frozen after your last `sim.run()` completes, so `disconnect()` cannot be processed. Simply restart your kernel when you're done with all experiments.

### macOS Best Practices

1. **Use `render=False` for development** - `close()` works perfectly without GUI
2. **Create simulation once** - don't create multiple `Simulation()` objects
3. **Don't call `close()` in GUI mode** - restart kernel when done
4. **Run all experiments before stopping** - plan your experiments to minimize restarts

## Files

- `sim_class_FINAL.py` - Recommended version (includes macOS fixes)
- `sim_class.py` - Original version
- `demo.ipynb` - Working examples
- `pybullet-macos-sequoia-fix.md` - Installation guide for macOS Sequoia

## Troubleshooting

**Window shows spinning wheel on startup?**  
Run the code - window becomes responsive after first `run()` call

**Window frozen between runs?**  
Normal on macOS - becomes responsive during next `run()` call

**Need to close window?**  
Use `render=False` for development, or restart kernel in Jupyter

**Multiple windows piling up?**  
Use `sim_class_FINAL.py` and create simulation once, reuse with `reset()`

## Credits

Fixes based on PyBullet Quickstart Guide and community issue reports:
- [PyBullet GUI freeze on macOS](https://github.com/bulletphysics/bullet3/issues/2014)
- [Clang 18 compilation issues](https://github.com/bulletphysics/bullet3/issues/4607)