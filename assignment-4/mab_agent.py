'''
  mab_agent.py
  
  Agent specifications implementing Action Selection Rules.
'''

import numpy as np

# ----------------------------------------------------------------
# MAB Agent Superclasses
# ----------------------------------------------------------------


class MAB_Agent:
    '''
    MAB Agent superclass designed to abstract common components
    between individual bandit players (below)
    '''

    def __init__(self, K):
        # TODO: Placeholder: add whatever you want here
        self.K = K
        self.tries = [0 for _ in range(self.K)]
        self.wins = [0 for _ in range(self.K)]

    def give_feedback(self, a_t, r_t):
        '''
        Provides the action a_t and reward r_t chosen and received
        in the most recent trial, allowing the agent to update its
        history
        '''
        self.tries[a_t] += 1
        self.wins[a_t] += r_t

    def clear_history(self):
        '''
        IMPORTANT: Resets your agent's history between simulations.
        No information is allowed to transfer between each of the N
        repetitions
        '''
        self.tries = [0 for _ in range(self.K)]
        self.wins = [0 for _ in range(self.K)]


# ----------------------------------------------------------------
# MAB Agent Subclasses
# ----------------------------------------------------------------

class Greedy_Agent(MAB_Agent):
    '''
    Greedy bandit player that, at every trial, selects the
    arm with the presently-highest sampled Q value
    '''

    def __init__(self, K):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        # TODO: Currently makes a random choice -- change!
        return np.random.choice(list(range(self.K)))


class Epsilon_Greedy_Agent(MAB_Agent):
    '''
    Exploratory bandit player that makes the greedy choice with
    probability 1-epsilon, and chooses randomly with probability
    epsilon
    '''

    def __init__(self, K, epsilon):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        # TODO: Currently makes a random choice -- change!
        return np.random.choice(list(range(self.K)))


class Epsilon_First_Agent(MAB_Agent):
    '''
    Exploratory bandit player that takes the first epsilon*T
    trials to randomly explore, and thereafter chooses greedily
    '''

    def __init__(self, K, epsilon, T):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        # TODO: Currently makes a random choice -- change!
        return np.random.choice(list(range(self.K)))


class Epsilon_Decreasing_Agent(MAB_Agent):
    '''
    Exploratory bandit player that acts like epsilon-greedy but
    with a decreasing value of epsilon over time
    '''

    def __init__(self, K):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        # TODO: Currently makes a random choice -- change!
        return np.random.choice(list(range(self.K)))


class TS_Agent(MAB_Agent):
    '''
    Thompson Sampling bandit player that self-adjusts exploration
    vs. exploitation by sampling arm qualities from successes
    summarized by a corresponding beta distribution
    '''

    def __init__(self, K):
        MAB_Agent.__init__(self, K)

    def choose(self, *args):
        # TODO: Currently makes a random choice -- change!
        return np.random.choice(list(range(self.K)))
