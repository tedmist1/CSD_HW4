# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game

import numpy as np
import pandas as pd

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

all_available_actions = [
    'Stop',
    'North',
    'East',
    'South',
    'West'
]

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    self.start = gameState.getAgentPosition(self.index)


    self.q_learn = QLearningTable(actions=list(range(len(all_available_actions))),
        learning_rate=0.05)

    self.previous_state = gameState
    self.previous_action = None



    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)
    # Can maybe use getAgentState for previous action?
    reward = 3
    if self.previous_action:
        self.q_learn.learn(str(self.previous_state), self.previous_action, reward, str(gameState))

    self.previous_state = str(gameState)

    action = self.q_learn.choose_action(str(gameState))
    choice = all_available_actions[action]
    self.previous_action = choice


    '''
    You should change this in your own agent.
    '''
    return choice

class QLearningTable:
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9, e_decay = 1):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_decay = e_decay
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        self.epsilon = 1 - ((1 - self.epsilon) * self.epsilon_decay)
        # At random chance, explore instead of exploit
        if np.random.uniform() < self.epsilon:
            # Get resulting Q values from State information
            state_action = self.q_table.ix[observation, :]
            # Gets max of the actions in the given state
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        # Current Q value for state action pair
        q_predict = self.q_table.ix[s, a]
        # Reward for performing action in the state, plus predicted reward for next state
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # Update Q value with the difference times the learning rate
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class AppxQLearningTable:
    def __init__(self, features, weights, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.features = features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.weights = weights

    def choose_action(self, observation):

        # At random chance, explore instead of exploit
        if np.random.uniform() < self.epsilon:
            # Get resulting Q values from State information
            state_action = self.q_table.ix[observation, :]
            # Gets max of the actions in the given state
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def estimate_state_action(self, function_val):
        sum = 0
        if len(function_val) is not len(self.weights):
            # Error
            return "Error"

        for i in range(len(weights)):
            sum += function_val[i] * self.weights[i]


    def learn(self, s, a, r, s_):

        # difference = r +

        # Current Q value for state action pair
        q_predict = self.q_table.ix[s, a]
        # Reward for performing action in the state, plus predicted reward for next state
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()

        # Update Q value with the difference times the learning rate
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)
