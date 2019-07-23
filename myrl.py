from mygridworld import GridWorldMDP

import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(utility_grids,solver_name):
    fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility with {}'.format(solver_name), color='b')

    # policy_changes = np.count_nonzero(np.diff(policy_grids[:,:,:end_step]), axis=(0, 1))
    # ax2.plot(policy_changes, 'r.-')
    # ax2.set_ylabel('Change in Best Policy', color='r')


def plot_difference(ori_utility_grid,utility_grids,solver_name):

    ori_utility_grid = np.expand_dims(ori_utility_grid,axis=-1)
    difference = np.sum(np.square(ori_utility_grid-utility_grids),axis=(0,1))
    plt.plot(difference,'b.-')
    plt.ylabel('Difference with true value grid with {}'.format(solver_name),color='b')

def plot_reward(reward_list):
    plt.xlabel('Episodes')
    plt.ylabel('Reward per Episode')
    # plt.xticks()
    plt.plot(reward_list,'b.-')





if __name__ == '__main__':
    shape = (4, 12)
    goal_list = [(-1,-1)]
    start_list = [(3,0)]
    cliff_list = []
    for i in range(10):
        cliff_list.append((3,i+1))

    default_reward = -1
    cliff_reward = -100
    reward_grid = np.zeros(shape) + default_reward
    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)
    cliff_mask = np.zeros_like(reward_grid,dtype= np.bool)
    for i in range(len(goal_list)):
        terminal_mask[goal_list[i]] = True
    for i in range(len(cliff_list)):
        cliff_mask[cliff_list[i]] = True

    indices = np.arange(shape[0]*shape[1])
    r0,c0 = np.unravel_index(indices,shape)
    # reward_grid[r0[terminal_mask.flatten()],c0[terminal_mask.flatten()]] = 0
    cliff_locs = np.where(cliff_mask.flatten())[0]
    reward_grid[r0[cliff_locs],c0[cliff_locs]] = cliff_reward

    gw = GridWorldMDP(start_list = start_list,
                      reward_grid=reward_grid,
                      terminal_mask=terminal_mask,
                      cliff_mask = cliff_mask,
                      action_probabilities=[
                      # the probabilities of action mean possible other steps when taking accurate steps
                          (-1, 0),
                          (0, 1.0),
                          (1, 0),
                      ],
                      no_action_probability=0.0,
                      accuracy= 1e-5)

    # mdp_solvers = {'first_visit_MC':gw.firstv_MC, 'every_visit_MC':gw.everyv_MC,'Temporal_difference':gw.TD_0}
    # policy_grids, optimal_utility_grids, end_step = gw.run_policy_iterations(iterations=100, discount=1.0)
    # print("Final result of Policy_iteration : ")
    # print(optimal_utility_grids[:,:,end_step])
    # plt.figure()
    # gw.plot_policy(optimal_utility_grids[:,:,end_step])
    # plt.savefig("basic_policy_{}_{}.png".format(shape,"policy_iteration"))
    # plt.show()
    mdp_solvers = {'SARSA':gw.SARSA,'Q_learning':gw.Q_learning}

    for solver_name, solver_fn in mdp_solvers.items():

        # plt.figure()
        # plot_convergence(optimal_utility_grids[:,:,:end_step])
        # plt.show()
        epsilon_list = [0,0.05,0.1,0.15,0.2]
        # epsilon_list = [0.15]
        episode = 5000
        for ep in epsilon_list:
            policy_grid,utility_grid,reward_list = solver_fn(delta = 1e-2,discount = 1.0,epsilon = ep,step_size = 0.01, episode = episode)
            # every_utility_grids = gw.firstv_MC(iterations=1000, discount=1.0, policy_grid=policy_grids[:, :, end_step])
            # utility_grids = np.array(utility_grids)
            print('Final result of {}:'.format(solver_name))
            # utility_grids = solver_fn(discount=1)
            # print(policy_grids[:, :, end_step])
            # print(utility_grids[:,:,-1])
            print(utility_grid)
            print(policy_grid)
            print(reward_list)
            plt.figure()
            gw.plot_policy(utility_grid,policy_grid)
            plt.savefig('policy_{}_{}_{}_decay.png'.format(solver_name,ep,episode))
            plt.show()

            plot_reward(reward_list)
            plt.savefig('reward_plot_{}_{}_{}_decay.png'.format(solver_name,ep,episode))
            plt.show()

        # plot_convergence(utility_grids,solver_name)
        # plt.savefig('u&p_change_curve_{}_{}.png'.format(shape,solver_name))
        # plt.show()

        # plot_difference(optimal_utility_grids[:,:,end_step],utility_grids,solver_name)
        # plt.savefig('value_ssd_about_true_value_grid_{}_{}.png'.format(shape,solver_name))
        # plt.show()
