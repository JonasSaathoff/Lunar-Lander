# GitHub Copilot Instructions

## Project Overview
This is a Natural Computing (NACO) assignment implementing optimization algorithms for the Lunar Lander problem using OpenAI Gymnasium. The core architecture consists of:

- **`problem.py`**: `GymProblem` wrapper class around Gymnasium's LunarLander-v3 environment
- **`random_search.py`**: Example implementation of random search optimization algorithm
- **Assignment goal**: Implement and compare different optimization algorithms (genetic algorithms, particle swarm, etc.)

## Key Architecture Patterns

### Solution Representation
Solutions are represented as flattened numpy arrays of shape `(state_size * n_outputs,)` that get reshaped into control matrices:
```python
# Solutions are flattened: x.shape = (state_size * n_outputs,)
self.M = x.reshape(self.state_size, self.n_outputs)  # Control matrix
action = self.activation(self.M.T @ observation)     # Linear control policy
```

### Environment Configuration
The `GymProblem` class supports both discrete and continuous action spaces:
- **Discrete**: Uses `np.argmax` activation, `n_outputs = env.action_space.n`
- **Continuous**: Uses `np.tanh` activation, `n_outputs = env.action_space.shape[0]`

### Evaluation Pattern
All optimization algorithms should follow this evaluation pattern:
```python
problem = GymProblem()  # Initialize environment
x = problem.sample()    # Generate/mutate solution
f, rewards = problem(x) # Evaluate fitness (returns total reward + reward history)
problem.show(x)         # Visualize best solution
```

## Development Workflows

### Dependencies
```bash
pip install "gymnasium[box2d]" numpy matplotlib
```

### Running Algorithms
```bash
python random_search.py  # Run example random search
python your_algorithm.py # Your optimization implementation
```

### Testing Solutions
- Use `problem.play_episode(x)` for headless evaluation
- Use `problem.show(x)` to visualize specific solutions with rendering
- Track both fitness values and reward trajectories for analysis

## Implementation Guidelines

### New Optimization Algorithms
When implementing new algorithms (GA, PSO, etc.):
1. Import `GymProblem` from `problem.py`
2. Use `problem.sample()` for initial population/random solutions
3. Solutions must be numpy arrays in range `[-1, 1]`
4. Follow the evaluation loop pattern from `random_search.py`
5. Plot both fitness history and best solution rewards

### Problem Configuration
Key `GymProblem` parameters for experiments:
- `continuous=True/False`: Action space type
- `gravity=-10.0`: Moon gravity (default), can vary for difficulty
- `enable_wind=True, wind_power=15.0`: Add environmental challenges
- `turbulence_power=1.5`: Add stochastic disturbances

### Performance Considerations
- Episodes auto-terminate on landing/crash (max 1000 steps)
- Fitness is cumulative reward (higher = better landing)
- Use `budget` parameter to limit function evaluations for fair comparison
- Cache best solutions for visualization and analysis

## File Organization
- Keep algorithm implementations in separate files (`genetic_algorithm.py`, `pso.py`, etc.)
- Don't modify `problem.py` (assignment constraint)
- Follow the plotting pattern from `random_search.py` for consistent result visualization