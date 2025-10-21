'''
Requirements: 
    pip install "gymnasium[box2d]" numpy matplotlib
'''

from problem import GymProblem
import numpy as np
import matplotlib.pyplot as plt

def main():
    # generate a problem instance with default simulation parameters (no wind, no turbulences and default moon gravity)
    problem = GymProblem()

    budget = 1000

    f_hist = []
    f_prime = -np.inf
    x_prime = None
    best_rewards = None

    for i in range(budget):
        x = problem.sample() # generate a random solution

        f, rewards = problem.play_episode(x) # evaluate the solution
        
        if f > f_prime: # if improvement, save the best solution found
            f_prime = f
            x_prime = x.copy()
            best_rewards = rewards

        f_hist.append(f)
        print(f"Gen {i}: Best f = {f_prime:.3f}")

    problem.show(x_prime) # show the best solution found

    # subplot with f history and rewards of the best solution
    plt.subplot(2, 1, 1)
    plt.plot(f_hist)
    plt.title("Fitness History")
    plt.xlabel("Itteration")
    plt.ylabel("Fitness")
    plt.subplot(2, 1, 2)
    plt.plot(best_rewards)
    plt.title("Best solution rewards")
    plt.xlabel("Simulation step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
