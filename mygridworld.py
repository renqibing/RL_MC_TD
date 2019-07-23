import numpy as np
import matplotlib.pyplot as plt
import cv2


class GridWorldMDP:

    # up, right, down, left
    # left, down, right, up
    _direction_deltas = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]

    _num_actions = len(_direction_deltas)

    def __init__(self,
                 start_list,
                 reward_grid,
                 terminal_mask,
                 action_probabilities,
                 no_action_probability,
                 cliff_mask,
                 accuracy
                 ):
        self.start_list = start_list
        self._cliff_mask = cliff_mask
        self._reward_grid = reward_grid
        self._terminal_mask = terminal_mask
        self._T = self._create_transition_matrix(
            action_probabilities,
            no_action_probability
        )
        self.accuracy = accuracy
        # self.gamma = gamma
    @property
    def shape(self):
        return self._reward_grid.shape

    @property
    def list_shape(self):
        return list(self.shape)

    @property
    def size(self):
        return self._reward_grid.size

    @property
    def reward_grid(self):
        return self._reward_grid

    def run_value_iterations(self, discount=1.0,
                             iterations=10):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)
        utility_grid = np.zeros_like(self._reward_grid)
        # bias = 0.0
        for i in range(iterations):
            # original_grid = utility_grid.copy()
            utility_grid = self._value_iteration(utility_grid=utility_grid)
            # bias = max((bias,np.absolute(np.max(utility_grid-original_grid))))
            policy_grids[:, :, i] = self.best_policy(utility_grid)
            utility_grids[:, :, i] = utility_grid
            # if bias < 1e-4:
            #     break
        return policy_grids, utility_grids

    def run_policy_iterations(self, discount=1.0,
                              iterations=10):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)
        # policy_dims = self.list_shape().append(4)
        M,N = self.shape
        policy_grid = np.zeros((M,N,self._num_actions)) + 1.0/self._num_actions
        utility_grid = self._reward_grid.copy()
        # bias = 0.0
        bias_list = []
        for i in range(iterations):
            original_grid = utility_grid.copy()
            policy_grid, utility_grid = self._policy_iteration(
                policy_grid=policy_grid,
                utility_grid=utility_grid,
                discount=discount
            )
            if i == 0:
                bias = np.max(np.absolute(utility_grid - original_grid))
            else:
                bias = min(bias, np.max(np.absolute(utility_grid - original_grid)))
            bias_list.append(bias)
            policy_grids[:, :, i] = np.argmax(policy_grid,axis=2)
            utility_grids[:, :, i] = utility_grid
            if bias < self.accuracy:
                break
        end_step = i
        self.draw_bias(bias_list)
        return policy_grids, utility_grids,end_step

    def draw_bias(self,bias_list):
        plt.figure()
        plt.plot(bias_list,'b.-')
        plt.title('Change in accuracy between neighbor nodes')
        plt.xlabel('Iteration step')
        plt.ylabel('Change in accuracy')
        plt.savefig('bias_change_curve_{}.png'.format(self.shape))
        plt.show()


    def generate_experience(self, current_state_idx,action_idx):
        sr, sc = self.grid_indices_to_coordinates(current_state_idx)
        next_state_probs = self._T[sr, sc, action_idx, :, :].flatten()

        next_state_idx = np.random.choice(np.arange(next_state_probs.size),
                                          p=next_state_probs)

        return (current_state_idx,next_state_idx,
                self._reward_grid.flatten()[next_state_idx],
                self._terminal_mask.flatten()[next_state_idx])

    def generate_episode(self,policy_grid):
        episode = []
        start_idx = np.random.randint(self._reward_grid.flatten().shape[0])
        # sr,sc = self.grid_indices_to_coordinates(start_idx)
        terminal_flag = False
        current_idx = start_idx
        while not terminal_flag:
            # act_idx = np.random.randint(self._num_actions)
            sr,sc = self.grid_indices_to_coordinates(current_idx)
            act_idx = policy_grid[sr,sc]
            exp = self.generate_experience(current_idx,int(act_idx))
            terminal_flag = exp[-1]
            current_idx = exp[1]
            episode.append(exp)
        # for i in range(len(episode)):
        #     print('{} to {}'.format(self.grid_indices_to_coordinates(episode[i][0]),self.grid_indices_to_coordinates(episode[i][1])))
        return episode

    def firstv_MC(self,iterations,discount,policy_grid): # exp next_state_idx reward
        theta = self.accuracy
        M,N = self.shape
        utility_grid = np.zeros((M, N))
        visited_grid = np.zeros((M, N)).flatten()
        num_visit = np.zeros(M*N)
        last_utility = np.zeros((M, N))
        utility_grids = np.zeros((M,N,iterations))
        for i in range(iterations):
            visited_grid[:] = 0
            exp = self.generate_episode(policy_grid)
            length = len(exp)
            G_t = 0.0
            for t_i in range(length):
                re_t_i = length - t_i - 1
                cs_i, ns_i, reward = exp[re_t_i][:3]
                G_t = reward + discount * G_t
                # print("return in {t} time is {g}".format(t = re_t_i,g= G_t))
                if (visited_grid[cs_i]):
                    continue
                else:
                    visited_grid[cs_i] = True
                num_visit[cs_i] += 1
                sr,sc = self.grid_indices_to_coordinates(cs_i)
                utility_grid[sr,sc] += (G_t - utility_grid[sr,sc] ) / num_visit[cs_i]
            utility_grids[:,:,i] = utility_grid
        return utility_grids

    def everyv_MC(self,iterations,discount,policy_grid): # exp next_state_idx reward
        theta = self.accuracy
        M,N = self.shape
        utility_grid = np.zeros((M, N))
        # bias = 1.0
        visited_grid = np.zeros((M, N)).flatten()
        num_visit = np.zeros(M*N)
        last_utility = np.zeros((M, N))
        utility_grids = np.zeros((M,N,iterations))
        for i in range(iterations):
            exp = self.generate_episode(policy_grid)
            length = len(exp)
            G_t = 0.0
            for t_i in range(length):
                re_t_i = length - t_i - 1
                cs_i, ns_i, reward = exp[re_t_i][:3]
                G_t = reward + discount * G_t
                # print("return in {t} time is {g}".format(t = re_t_i,g= G_t))
                num_visit[cs_i] += 1
                sr,sc = self.grid_indices_to_coordinates(cs_i)
                utility_grid[sr,sc] += (G_t - utility_grid[sr,sc] ) / num_visit[cs_i]
            utility_grids[:,:,i] = utility_grid
        return utility_grids

    def TD_0(self,iterations,discount,policy_grid):
        theta = self.accuracy
        M, N = self.shape
        utility_grid = np.zeros((M, N))
        # bias = 1.0
        visited_grid = np.zeros((M, N)).flatten()
        num_visit = np.zeros(M * N)
        last_utility = np.zeros((M, N))
        utility_grids = np.zeros((M, N, iterations))
        for i in range(iterations):
            # visited_grid[:] = 0
            exp = self.generate_episode(policy_grid)
            length = len(exp)
            # G_t = 0.0
            for t_i in range(length):
                re_t_i = length - t_i - 1
                cs_i, ns_i, reward = exp[re_t_i][:3]
                sr, sc = self.grid_indices_to_coordinates(cs_i)
                nr,nc = self.grid_indices_to_coordinates(ns_i)
                temporal_return = reward + discount*utility_grid[nr,nc]
                # G_t = reward + discount * G_t
                num_visit[cs_i] += 1
                utility_grid[sr, sc] += (temporal_return - utility_grid[sr, sc]) / num_visit[cs_i]
            utility_grids[:, :, i] = utility_grid

        return utility_grids

    def SARSA(self,delta,discount,epsilon,step_size,episode):
        '''
        # first initialize q(s,a) r(s) v(s)
        # repeat until termial flag
        # second choose greedy action from epison-greedy algorithm
        # third update q(s,a) within the current policy
        # update the next s until convergence
        discount represents coefficients of discounted return G_t
        epsilon represents e-greedy
        delta shows when to converge
        '''

        sse = 10
        M, N = self.shape
        A = len(self._direction_deltas)
        Q_grid = np.random.rand(M,N,A)
        act_prob_list = np.zeros(A) + epsilon/A
        act_list = np.arange(A)
        # policy_grid = np.zeros(M,N)
        for mask in self._terminal_mask:
            Q_grid[mask[0],mask[1],:] = 0
        terminal_flag = False
        reward_list = []
        step = 0
        ori_epsilon = epsilon
        while step < episode:
            step += 1
            if step % 100 :
                epsilon /= 2
            act_prob_list[:] = epsilon / A
            # init_s = np.random.randint(self._reward_grid.flatten().shape[0]-1)
            # choose epsilon-greedy action

            # start from the initial state
            sx,sy = self.start_list[0]
            opt_index = np.argmax(Q_grid[sx, sy, :])
            act_prob_list[opt_index] += 1 - epsilon
            cur_act_index = np.random.choice(act_list, p=act_prob_list)
            ori_Q_grid = Q_grid.copy()
            # reward per episode
            episode_r = 0

            while not terminal_flag:
                # GET S' and R from S and A
                # print("sx:{} sy:{}".format(sx,sy))
                next_s_prob = self._T[sx,sy,cur_act_index,:,:].flatten()
                # print("sx {}sy {} cur_act_index {} sum of probility {}".format(sx,sy,cur_act_index,np.sum(next_s_prob)))
                next_s = np.random.choice(np.arange(next_s_prob.size),p= next_s_prob)
                # print("next_s_prob")
                cur_r = self._reward_grid.flatten()[next_s]
                episode_r += cur_r

                # Choose Next E-greedy step
                nsx, nsy = self.grid_indices_to_coordinates(next_s)
                opt_index = np.argmax(Q_grid[nsx, nsy, :])
                # print("opt_index",opt_index)
                act_prob_list[:] = epsilon / A
                act_prob_list[opt_index] += 1 - epsilon
                next_act_index = np.random.choice(act_list, p=act_prob_list)
                # print(next_act_index)
                # update Q_grid Q(s,a) += a * (r + gamma * Q(s',a') - Q(s,a))
                Q_grid[sx,sy,cur_act_index] += step_size * (cur_r + discount * Q_grid[nsx,nsy,next_act_index] - Q_grid[sx,sy,cur_act_index])

                cur_act_index = next_act_index
                sx = nsx
                sy = nsy
                # print("self._cliff_mask",self._cliff_mask[sx,sy])
                # if entering into the cliff or reaching the terminal, start a new episode
                terminal_flag = self._terminal_mask[sx,sy] or self._cliff_mask[sx,sy]
                # print(terminal_flag)
            terminal_flag = False
            reward_list.append(episode_r)
            print(episode_r)
            # sse = np.sum(np.square((Q_grid-ori_Q_grid)),axis=(0,1,2))
            # print("sse",sse)
        policy_grid = np.argmax(Q_grid,axis=-1)
        utility_grid = np.max(Q_grid,axis=-1)
        return policy_grid,utility_grid,reward_list

    def Q_learning(self,delta,discount,epsilon,step_size,episode):
        sse = 10
        M, N = self.shape
        A = len(self._direction_deltas)
        Q_grid = np.random.rand(M, N, A)
        act_prob_list = np.zeros(A) + epsilon / A
        act_list = np.arange(A)
        # policy_grid = np.zeros(M,N)
        for mask in self._terminal_mask:
            Q_grid[mask[0], mask[1], :] = 0
        terminal_flag = False
        ori_epsilon = epsilon
        step = 0
        reward_list = []
        while step < episode:
            step += 1
            if step % 100:
                epsilon /= 2
            act_prob_list[:] = epsilon / A
            # init_s = np.random.randint(self._reward_grid.flatten().shape[0] - 1)
            sx,sy = self.start_list[0]
            # ori_Q_grid = Q_grid.copy()
            episode_r = 0
            while not terminal_flag:
                # choose epsilon-greedy action
                act_prob_list[:] = epsilon / A
                # sx, sy = self.grid_indices_to_coordinates(init_s)
                opt_index = np.argmax(Q_grid[sx, sy, :])
                act_prob_list[opt_index] += 1 - epsilon
                cur_act_index = np.random.choice(act_list, p=act_prob_list)
                # get next state and reward
                next_s_prob = self._T[sx, sy, cur_act_index, :, :].flatten()
                next_s = np.random.choice(np.arange(next_s_prob.size), p=next_s_prob)
                cur_r = self._reward_grid.flatten()[next_s]
                episode_r += cur_r
                nsx, nsy = self.grid_indices_to_coordinates(next_s)
                # update Q_grid
                Q_grid[sx, sy, cur_act_index] += step_size * (
                            cur_r + discount * np.max(Q_grid[nsx,nsy,:]) - Q_grid[sx, sy, cur_act_index])
                sx = nsx
                sy = nsy
                terminal_flag = self._terminal_mask[sx, sy] or self._cliff_mask[sx,sy]
            terminal_flag = False
            reward_list.append(episode_r)
            print(episode_r)
            # sse = np.sum(np.square((Q_grid - ori_Q_grid)), axis=(0, 1, 2))
        policy_grid = np.argmax(Q_grid, axis=-1)
        utility_grid = np.max(Q_grid, axis=-1)
        return policy_grid, utility_grid,reward_list

    def grid_indices_to_coordinates(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coordinates_to_indices(self, coordinates=None):
        # Annoyingly, this doesn't work for negative indices.
        # The mode='wrap' parameter only works on positive indices.
        if coordinates is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coordinates, self.shape)

    def best_policy(self, utility_grid):
        M, N = self.shape
        greedy_index = np.argmax((utility_grid.reshape(1,1,1,M,N)*self._T).sum(axis=-1).sum(axis = -1),axis=2)
        greedy_one_hot = (np.arange(0,self._num_actions)==np.expand_dims(greedy_index,axis=-1)).astype(np.float32)

        return greedy_one_hot

    def _init_utility_policy_storage(self, depth):
        M, N = self.shape
        utility_grids = np.zeros((M, N, depth))
        policy_grids = np.zeros((M,N,depth))

        return utility_grids, policy_grids

    def _create_transition_matrix(self,
                                  action_probabilities,
                                  no_action_probability):
        M, N = self.shape

        T = np.zeros((M, N, self._num_actions, M, N))

        r0, c0 = self.grid_indices_to_coordinates()  # generate all coordinates of states from (0,0) to (-1,-1)

        T[r0, c0, :, r0, c0] += no_action_probability

        for action in range(self._num_actions):
            for offset, P in action_probabilities:
                direction = (action + offset) % self._num_actions  # possible actions except the required action

                dr, dc = self._direction_deltas[direction]  # the difference between the last action and the next action
                r1 = np.clip(r0 + dr, 0, M - 1)  # clip coor into the scale of the grid
                c1 = np.clip(c0 + dc, 0, N - 1)

                # temp_mask = obstacle_mask[r1, c1].flatten()
                # r1[temp_mask] = r0[temp_mask]
                # c1[temp_mask] = c0[temp_mask]

                T[r0, c0, action, r1, c1] += P

        terminal_locs = np.where(self._terminal_mask.flatten())[0]  # return indices of the terminal grid
        cliff_locs = np.where(self._cliff_mask.flatten())[0]
        T[r0[terminal_locs], c0[terminal_locs], :, :, :] = 0
        T[r0[terminal_locs],c0[terminal_locs],:,r0[terminal_locs],c0[terminal_locs]] = 1

        T[r0[cliff_locs],c0[cliff_locs], :, :, :] = 0
        T[r0[cliff_locs],c0[cliff_locs], :, self.start_list[0][0],self.start_list[0][1]] = 1

        return T

    def _value_iteration(self, utility_grid, discount=1.0):
        out = np.zeros_like(utility_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self._calculate_utility((i, j),
                                                    discount,
                                                    utility_grid)
        return out

    def _policy_iteration(self, *, utility_grid,
                          policy_grid, discount=1.0):
        r, c = self.grid_indices_to_coordinates()
        M, N = self.shape
        # v_s = sum_a * \pi_a * ( R_s^a + \lambda * \sum_s' P_ss' * v_s' )
        # policy_grid  (shape,num_action,)
        utility_grid = (
            self._reward_grid +
            discount * np.sum(np.squeeze((utility_grid.reshape((1, 1, 1, M, N)) * self._T)
                        .sum(axis=-1).sum(axis=-1))*policy_grid,axis=-1)
            .reshape(self.shape)
        )

        utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]

        return self.best_policy(utility_grid), utility_grid

    def _calculate_utility(self, loc, discount, utility_grid):
        if self._terminal_mask[loc]:
            return self._reward_grid[loc]
        row, col = loc
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, :, :] * utility_grid,
                       axis=-1),
                axis=-1)
        ) + self._reward_grid[loc]

    def plot_policy(self, utility_grid, policy_grid=None):
        if policy_grid is None:
            policy_grid = self.best_policy(utility_grid)
        markers = "^>v<"
        marker_size = 200 // np.max(policy_grid.shape)
        marker_edge_width = marker_size // 100
        marker_fill_color = 'w'

        no_action_mask = self._terminal_mask

        utility_normalized = (utility_grid - utility_grid.min()) / \
                             (utility_grid.max() - utility_grid.min())

        utility_normalized = (255*utility_normalized).astype(np.uint8)

        utility_rgb = cv2.applyColorMap(utility_normalized, cv2.COLORMAP_JET)
        for i in range(3):
            channel = utility_rgb[:, :, i]
            # channel[self._obstacle_mask] = 0

        plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')
        # policy_grid = np.argmax(policy_grid,axis=2)
        for i, marker in enumerate(markers):
            y, x = np.where((policy_grid == i) & np.logical_not(no_action_mask))
            plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

        y, x = np.where(self._terminal_mask)
        plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
                 color=marker_fill_color)

        tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
        tick_step = np.max(policy_grid.shape)/8
        best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
        tick_step = tick_step_options[best_option]
        plt.xticks(np.arange(0, policy_grid.shape[1] - 0.5, tick_step))
        plt.yticks(np.arange(0, policy_grid.shape[0] - 0.5, tick_step))
        plt.xlim([-0.5, policy_grid.shape[0]-0.5])
        plt.xlim([-0.5, policy_grid.shape[1]-0.5])

