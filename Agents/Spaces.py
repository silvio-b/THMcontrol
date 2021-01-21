import itertools
import pandas as pd


class DiscreteSpace:
    def __init__(self, breakpoints):
        self.breakpoints = breakpoints
        self.n_var = len(self.breakpoints)
        self.states = [list(range(0, len(x) - 1)) for x in self.breakpoints]
        self.comb_states = list(itertools.product(*self.states))
        self.space_dim = len(self.comb_states)

    def get_index(self, values):
        index = []
        for s in range(0, self.n_var):
            pos = [x - values[s] for x in self.breakpoints[s]]
            index.append(pos.index(max([i for i in pos if i < 0])))
        index = self.comb_states.index(tuple(index))
        return index

    def get_space_table(self):

        columns = ['{}{}'.format(ch, x) for x in range(self.n_var) for ch in ['min', 'max']]
        columns = ['state'] + columns
        space_table = pd.DataFrame(columns=columns)
        for i in range(0, len(self.comb_states)):
            sp_tbl_i = [str(self.comb_states[i])]
            for j in range(0, len(self.comb_states[i])):
                sp_tbl_i.append(self.breakpoints[j][self.comb_states[i][j]])
                sp_tbl_i.append(self.breakpoints[j][self.comb_states[i][j] + 1])
            # print(sp_tbl_i)
            sp_tbl_i = pd.DataFrame([sp_tbl_i], columns=columns)
            space_table = pd.concat([space_table, sp_tbl_i])
        space_table.reset_index(inplace=True, drop=True)

        return space_table


