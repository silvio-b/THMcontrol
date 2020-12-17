import numpy as np


class TabularQLearning:

    def __init__(self, lr, gamma, action_space_shape, state_space_shape):
        self.lr = lr
        self.gamma = gamma
        self.action_space_shape = action_space_shape
        self.state_space_shape = state_space_shape

        self.q_table = np.zeros((self.state_space_shape, self.action_space_shape))
        # self.q_table = np.random.randn(self.state_space_shape, self.action_space_shape)

    def get_greedy_action(self, state_index):

        action = np.argmax(self.q_table[state_index])

        return action

    def update_table(self, state_index, action_index, next_state_index, reward):

        old_q_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index])

        new_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + self.gamma * next_max)

        self.q_table[state_index, action_index] = new_q_value