# 2526 NACO assignment: Landing a ship on the moon

This directory holds the implementation of a wrapper around the Lunar Lander problem from [gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/) and a simple random search to give an example on how to use it.


## GymProblem
This class is the wrapper around the problem, you are not supposed to edit this code. Here is a quick explaination of the different methods you might use:

### Initialisation
Initialisation example:
```python
GymProblem(
    continuous: bool = False,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 0,
    turbulence_power: float = 0
)
```
Parameters:
- `continuous` (bool): If True, use continuous actions; otherwise use discrete.
- `gravity` (float): Gravity setting for the environment (default -10.0).
- `enable_wind` (bool): Whether wind effects are enabled.
- `wind_power` (float): Strength of the wind effect.
- `turbulence_power` (float): Strength of turbulence.

## `.sample()`
Creates a randomly generated solution from $\mathcal{U}^n\sim[-1,1]$

### `.play_episode(x)`
Runs the simulation using candidate `x`. Returns the fitness and a list of all the rewards measured.


## `.show(x)`
Runs the simulation using candidate x with rendering enabled.

## Example usage:
```python
import numpy as np

# Create a discrete LunarLander problem
problem = GymProblem(continuous=False)

# Generate random candidate
x = problem.sample()

# Run an episode without rendering
f, rewards = problem(x)
print("Fitness: ", f)
print("Measured rewards: ", rewards)

# Render the landing
problem.show(x)
```