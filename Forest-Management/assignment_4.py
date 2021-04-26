import numpy as np
import time
import matplotlib.pyplot as plt
import mdptoolbox, mdptoolbox.example


def Forest_Management(state):
    compute_policy_iteration(state)

    compute_value_iteration(state)

    compute_q_learning(state)


def compute_q_learning(state):
    print("Q LEARNING WITH FOREST MANAGEMENT")
    P, R = mdptoolbox.example.forest(S=state, p=0.01)
    value_f = []
    policy = []
    iters = []
    time_array = []
    Q_table = []
    reward_array = []
    for epsilon in [0.05, 0.15, 0.25, 0.5, 0.75, 0.95]:
        st = time.time()
        pi = mdptoolbox.mdp.QLearning(P, R, 0.95)
        end = time.time()
        pi.run()
        reward_array.append(pi.reward)
        value_f.append(np.mean(pi.V))
        policy.append(pi.policy)
        time_array.append(end - st)
        Q_table.append(pi.Q)
    plt.plot(range(0, 10000), reward_array[0], label="epsilon=0.05")
    plt.plot(range(0, 10000), reward_array[1], label="epsilon=0.15")
    plt.plot(range(0, 10000), reward_array[2], label="epsilon=0.25")
    plt.plot(range(0, 10000), reward_array[3], label="epsilon=0.50")
    plt.plot(range(0, 10000), reward_array[4], label="epsilon=0.75")
    plt.plot(range(0, 10000), reward_array[5], label="epsilon=0.95")
    plt.legend()
    plt.xlabel("Iterations")
    plt.grid()
    plt.title("Forest Management - Q Learning - Decaying Epsilon")
    plt.ylabel("Average Reward")
    plt.grid()
    plt.savefig()
    plt.subplot(1, 6, 1)
    plt.imshow(Q_table[0][:20, :])
    plt.title("Epsilon=0.05")
    plt.subplot(1, 6, 2)
    plt.title("Epsilon=0.15")
    plt.imshow(Q_table[1][:20, :])
    plt.subplot(1, 6, 3)
    plt.title("Epsilon=0.25")
    plt.imshow(Q_table[2][:20, :])
    plt.subplot(1, 6, 4)
    plt.title("Epsilon=0.50")
    plt.imshow(Q_table[3][:20, :])
    plt.subplot(1, 6, 5)
    plt.title("Epsilon=0.75")
    plt.imshow(Q_table[4][:20, :])
    plt.subplot(1, 6, 6)
    plt.title("Epsilon=0.95")
    plt.imshow(Q_table[5][:20, :])
    plt.colorbar()
    plt.grid()
    plt.savefig("Forest Management - Q Learning - Decaying Epsilon.png")


def compute_value_iteration(state):
    print("VALUE ITERATION WITH FOREST MANAGEMENT")
    P, R = mdptoolbox.example.forest(S=state)
    value_f = [0] * 10
    policy = [0] * 10
    iters = [0] * 10
    time_array = [0] * 10
    gamma_arr = [0] * 10
    for i in range(0, 10):
        pi = mdptoolbox.mdp.ValueIteration(P, R, (i + 0.5) / 10)
        pi.run()
        gamma_arr[i] = (i + 0.5) / 10
        value_f[i] = np.mean(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time
    plt.plot(gamma_arr, time_array)
    plt.xlabel("Gammas")
    plt.title("Forest Management - Value Iteration - Execution Time Analysis")
    plt.ylabel("Execution Time (s)")
    plt.grid()
    plt.savefig("Forest Management - Value Iteration - Execution Time Analysis.png")
    plt.plot(gamma_arr, value_f)
    plt.xlabel("Gammas")
    plt.ylabel("Average Rewards")
    plt.title("Forest Management - Value Iteration - Reward Analysis")
    plt.grid()
    plt.savefig("Forest Management - Value Iteration - Reward Analysis.png")
    plt.plot(gamma_arr, iters)
    plt.xlabel("Gammas")
    plt.ylabel("Iterations to Converge")
    plt.title("Forest Management - Value Iteration - Convergence Analysis")
    plt.grid()
    plt.savefig("Forest Management - Value Iteration - Convergence Analysis.png")


def compute_policy_iteration(state):
    print("POLICY ITERATION WITH FOREST MANAGEMENT")
    P, R = mdptoolbox.example.forest(S=state)
    value_f = [0] * 10
    policy = [0] * 10
    iters = [0] * 10
    time_array = [0] * 10
    gamma_arr = [0] * 10
    for i in range(0, 10):
        print(f'DR: {i}')
        pi = mdptoolbox.mdp.PolicyIteration(P, R, (i + 0.5) / 10)
        pi.run()
        gamma_arr[i] = (i + 0.5) / 10
        value_f[i] = np.mean(pi.V)
        policy[i] = pi.policy
        iters[i] = pi.iter
        time_array[i] = pi.time
    plt.plot(gamma_arr, time_array)
    plt.xlabel("Gammas")
    plt.title("Forest Management - Policy Iteration - Execution Time Analysis")
    plt.ylabel("Execution Time (s)")
    plt.grid()
    plt.savefig("Forest Management - Policy Iteration - Execution Time Analysis.png")
    plt.plot(gamma_arr, value_f)
    plt.xlabel("Gammas")
    plt.ylabel("Average Rewards")
    plt.title("Forest Management - Policy Iteration - Reward Analysis")
    plt.grid()
    plt.savefig("Forest Management - Policy Iteration - Reward Analysis.png")
    plt.plot(gamma_arr, iters)
    plt.xlabel("Gammas")
    plt.ylabel("Iterations to Converge")
    plt.title("Forest Management - Policy Iteration - Convergence Analysis")
    plt.grid()
    plt.savefig("Forest Management - Policy Iteration - Convergence Analysis.png")


def colors_lake():
    return {
        b"S": "green",
        b"F": "skyblue",
        b"H": "black",
        b"G": "gold",
    }


def directions_lake():
    return {3: "⬆", 2: "➡", 1: "⬇", 0: "⬅"}


def actions_taxi():
    return {0: "⬇", 1: "⬆", 2: "➡", 3: "⬅", 4: "P", 5: "D"}


def colors_taxi():
    return {b"+": "red", b"-": "green", b"R": "yellow", b"G": "blue", b"Y": "gold"}


print("STARTING EXPERIMENTS")
print("Experment 1 with 1000 states")
Forest_Management(state=1000)
print("Experiment 1 ENDS===========")
print("Experiment 2 with 5000 states")
state = 5000
Forest_Management(state=5000)
print("Experiment 2 ENDS===========")
print("END OF EXPERIMENTS")
