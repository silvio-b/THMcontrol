import numpy as np
import decimal

def float_range(start, stop, step):         #Decreasing RANGE
  while start > stop:
    yield float(start)
    start -= decimal.Decimal(step)

class TabularQLearning:

    def __init__(self, lr, gamma, eps, temperature, action_space_shape, state_space_shape, season):
        self.lr = lr
        self.gamma = gamma
        self.eps_hot = eps
        self.eps_cold = eps
        self.temp_hot = temperature
        self.temp_cold = temperature
        self.action_space_shape = action_space_shape
        self.state_space_shape = state_space_shape
        self.season = season    # 0: Cold/Winter; 1: Hot/Summer


        self.action_space = [i for i in range(0,self.action_space_shape)]

        self.q_table_cold = np.zeros((self.state_space_shape, self.action_space_shape))
        self.q_table_hot = np.zeros((self.state_space_shape, self.action_space_shape))
        # self.q_table = np.random.randn(self.state_space_shape, self.action_space_shape)

    def get_greedy_action(self, state_index):

        if self.season == 0:
            table = self.q_table_cold
            eps = self.eps_cold
            self.eps_cold = self.eps_cold - 0.005 * self.eps_cold
        elif self.season == 1:
            table = self.q_table_hot
            eps = self.eps_hot
            self.eps_hot = self.eps_hot - 0.005 * self.eps_hot
        else:
            print("ERROR: Wrong flag for agent's season feature")
            print("Season feature is set to: ", self.season)

        #action = np.argmax(table[state_index])     #NO eps-greedy

        p = np.random.random()                      #YES eps-greedy
        if p < eps:
            action = np.random.choice(4)
        else:
            action = np.argmax(table[state_index])

        return action

    def get_softmax_action(self, state_index):

#        '''This want to decrease the temperatures of the seasonal agents separately'''
#        if self.season == 0:
#            scores = self.q_table_cold[state_index]
#            self.temp_cold -= self.temp_cold + ( (0.5-self.temp_cold)/1000 )
#            temp = self.temp_cold
#        if self.season == 1:
#            scores = self.q_table_hot[state_index]
#            self.temp_hot -= self.temp_hot + ( (0.5-self.temp_hot)/1000 )
#            temp = self.temp_hot

        '''if not using the above code, use following line:'''
        temp = self.temp_cold

        scores = self.q_table_cold[state_index]
        e_scores = np.exp(scores/temp)
        probs = e_scores / e_scores.sum()
        action = np.random.choice(self.action_space, p=probs)

        return action

    def update_table(self, state_index, action_index, next_state_index, reward):

        if self.season == 0:
            old_q_value = self.q_table_cold[state_index, action_index]
            next_max = np.max(self.q_table_cold[next_state_index])

            new_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + self.gamma * next_max)

            self.q_table_cold[state_index, action_index] = new_q_value

        elif self.season == 1:
            old_q_value = self.q_table_hot[state_index, action_index]
            next_max = np.max(self.q_table_hot[next_state_index])

            new_q_value = (1 - self.lr) * old_q_value + self.lr * (reward + self.gamma * next_max)

            self.q_table_hot[state_index, action_index] = new_q_value
#        else:
#            print("ERROR: Wrong flag for agent's season feature")

