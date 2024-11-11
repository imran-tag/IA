import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Q = 500x6

def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    # Q-learning update rule
    Q[s, a] = Q[s, a] + alpha * (r + gamma * Q[int(sprime), int(np.argmax(Q[sprime]))] - Q[s, a])
    return Q


def epsilon_greedy(Q, s, epsilon):
    """
    This function implements the epsilon greedy algorithm.
    Takes as input the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        return env.action_space.sample()
    # Exploitation: choose the action with the highest Q-value
    return np.argmax(Q[s])


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    env.reset()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.7
    gamma = 0.7
    epsilon = 0.05
    n_epochs = 2000
    max_itr_per_epoch = 500
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilon=epsilon)
            Sprime, R, done, _, info = env.step(A)
            r += R
            Q = update_q_table(Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma)
            S=Sprime
            # Update state and put a stoping criteria
            if done:
                break


        #print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs
    plt.plot(rewards)
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title("Rewards in function of epochs")
    plt.show()

    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    # show the learned policy
    env = gym.make("Taxi-v3", render_mode="human")
    env.reset()
    env.render()
    for _ in range(max_itr_per_epoch):
        A = np.argmax(Q[S])
        Sprime, R, done, _, info = env.step(A)
        S = Sprime
        if done:
            break
    env.close()
