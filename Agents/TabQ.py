import numpy as np
import decimal

def float_range(start, stop, step):         #Decreasing RANGE
  while start > stop:
    yield float(start)
    start -= decimal.Decimal(step)

class TabularQLearning:

    def __init__(self, lr, gamma, action_space_shape, state_space_shape):
        self.lr = lr
        self.gamma = gamma
        self.action_space_shape = action_space_shape
        self.state_space_shape = state_space_shape

        self.action_space = [i for i in range(0, self.action_space_shape)]

        self.q_table = np.zeros((self.state_space_shape, self.action_space_shape))

        #self.q_table = np.random.randn(self.state_space_shape, self.action_space_shape) * 10

    def get_greedy_action(self, state_index):

        action = np.argmax(self.q_table[state_index])

        return action

    def get_e_greedy_action(self, state_index, eps):

        p = np.random.random()

        if p < eps:
            action = np.random.choice(self.action_space_shape)
        else:
            action = np.argmax(self.q_table[state_index])

        return action

    def get_softmax_action(self, state_index, temp):

#        '''This want to decrease the temperatures of the seasonal agents separately'''
#        if self.season == 0:
#            scores = self.q_table_cold[state_index]
#            self.temp_cold -= self.temp_cold + ( (0.5-self.temp_cold)/1000 )
#            temp = self.temp_cold
#        if self.season == 1:
#            scores = self.q_table_hot[state_index]
#            self.temp_hot -= self.temp_hot + ( (0.5-self.temp_hot)/1000 )
#            temp = self.temp_hot


        scores = self.q_table[state_index]
        norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-4)
        e_scores = np.exp(norm_scores/temp)
        #e_scores = np.exp(scores/temp)
        probs = e_scores / e_scores.sum()
        action = np.random.choice(self.action_space, p=probs)

        return action

    def update_table(self, state_index, action_index, next_state_index, reward):

        old_q_value = self.q_table[state_index, action_index]
        next_max = np.max(self.q_table[next_state_index])

        new_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + self.gamma * next_max)

        self.q_table[state_index, action_index] = new_q_value

