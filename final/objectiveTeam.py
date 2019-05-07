# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random
import time
import util
import sys
from game import Directions
import game
import math
from util import nearestPoint

#################
# Team creation #
#################


def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveObjectiveAgent', second='OffensiveObjectiveAgent'):
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
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


# Safe Territory
obj1 = (15, 1)
obj2 = (15, 4)
obj3 = (15, 7)
obj4 = (15, 11)
obj5 = (15, 14)
obj11 = (11, 3)
obj12 = (13, 6)
obj13 = (12, 10)
obj14 = (12, 13)

# Enemy Territory
obj6 = (19, 2)
obj7 = (18, 7)
obj8 = (20, 13)
obj9 = (25, 2)
obj10 = (26, 10)
objectives = {
    obj1: [obj6, obj11, obj5],
    obj2: [obj6, obj12],
    obj3: [obj7, obj13, obj12],
    obj4: [obj7, obj5, obj14, obj13],
    obj5: [obj8, obj4, obj14, obj1],
    obj6: [obj9, obj7, obj2, obj1],
    obj7: [obj6, obj10, obj8, obj4, obj3],
    obj8: [obj7, obj10, obj5],
    obj9: [obj10, obj6],
    obj10: [obj8, obj7, obj9],
    obj11: [obj1, obj12],
    obj12: [obj2, obj3, obj13, obj11],
    obj13: [obj3, obj4, obj14, obj12],
    obj14: [obj4, obj5, obj13],
}

##########
# Agents #
##########


class ObjectiveCaptureAgent(CaptureAgent):
    """
    A base class for objective agents that chooses objective-maximizing actions
    """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        self.prevObjective = None
        self.currentObjective = None
        self.frontierObjectives = [obj11, obj12, obj13, obj14]
        self.futureObjective = None

        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest Q(s,a).
        """

        for objective in self.frontierObjectives:
            if gameState.getAgentPosition(self.index) == objective:
                self.prevObjective = self.currentObjective
                self.currentObjective = objective
                self.frontierObjectives = objectives[self.currentObjective]
                self.futureObjective = None

        objValues = [self.evaluateObjective(
            gameState, o) for o in self.frontierObjectives]
        # print(objValues)
        maxObjValue = max(objValues)
        bestObjectives = [a for a, v in zip(
            self.frontierObjectives, objValues) if v == maxObjValue]
        self.futureObjective = random.choice(bestObjectives)
        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]





        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 2:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            bestActions =  [bestAction]

        # UPDATE WEIGHTS HERE
        # Need to estimate Q value of next state
        reward = 1
        val_next_state = 1
        self.updateWeights(reward, maxValue, val_next_state, self.getFeatures(gameState, bestActions[0]))


        return random.choice(bestActions)


    def updateWeights(self, reward, Q_this_state, Q_next_state, features):
        learning_rate = 0.05 # Will be fixed
        reward_decay = 0.99 # Will be fixed

        difference = reward + reward_decay * Q_next_state - Q_this_state
        # print(difference)
        for weight in self.weights:
            # num = num + learning rate * feature value of that weight
            '''print(weight)
            print(self.weights[weight])
            #self.weights[weight] = self.weights[weight] + learning_rate * features[weight] * difference
            print(self.weights[weight] + learning_rate * features[weight] * difference)
            print("=========================================")'''

        return {'successorScore': 100, 'distanceToFood': -1, 'objective': -10}

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'successorScore': 1.0}

    def evaluateObjective(self, gameState, objective):
        features = self.getObjectiveFeatures(gameState, objective)
        weights = self.getObjectiveWeights(gameState, objective)
        return features * weights

    def getObjectiveFeatures(self, gameState, objective):
        """
        Returns a counter of features for the state
        """
        prevFobj = self.futureObjective
        self.futureObjective = objective
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        action = random.choice(bestActions)
        successor = self.getSuccessor(gameState, action)
        sucPos = successor.getAgentState(self.index).getPosition()

        features = util.Counter()
        # features['numNearbyPellets'] = self.numberNearbyPellets(
        #     gameState, objective)
        if self.prevObjective == objective:
            features['reverse'] = 1
        if prevFobj != objective:
            features['change'] = 1
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition()
                     != None]
        if len(defenders) > 0:
            dists = [self.getMazeDistance(
                sucPos, a.getPosition()) for a in defenders]
            features['inDanger'] = 5 - min(dists) if min(dists) < 5 else 0

        features['spread'] = objective[0] - \
            15 if self.getAgentClass(gameState) == 'A' else 0 - objective[0]

        features['risk'] = self.getScore(successor) - objective[1]

        self.futureObjective = prevFobj

        return features

    def getObjectiveWeights(self, gameState, objective):
        """
        Normally, weights do not depend on the gamestate.  They can be either
        a counter or a dictionary.
        """
        return {'inDanger': -20, 'reverse': -10, 'change': -5, 'spread': 0, 'risk': -3}

    def getAgentClass(self, gameState):
        if self.getTeam(gameState)[0] == self.index:
            return 'A'
        else:
            return 'B'

    def numberNearbyPellets(self, gameState, objective, action='Stop', maxDist=2):
        successor = self.getSuccessor(gameState, action)
        myPos = successor.getAgentState(self.index).getPosition()
        foodList = self.getFood(successor).asList()

        y2 = objective[1]
        y1 = myPos[1]
        x2 = objective[0]
        x1 = myPos[0]
        count = 0
        for food in foodList:
            y0 = food[1]
            x0 = food[0]
            if self.getDistance(myPos, objective) == 0:
                continue
            distance = math.fabs((y2 - y1) * x0 - (x2 - x1) * y0 +
                                 x2 * y1 - y2 * x2) / self.getDistance(myPos, objective)
            # print(distance)
            if distance < maxDist:
                count += 1

        return count

    def getDistance(self, pointA, pointB):
        return math.sqrt((pointB[0] - pointA[0]) * (pointB[0] - pointA[0]) +
                         (pointB[1] - pointA[1]) * (pointB[1] - pointA[1]))


class OffensiveObjectiveAgent(ObjectiveCaptureAgent):
    """
    An objective agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """
    fileName = 'offensive_save'
    weights = {'successorScore': 100, 'distanceToFood': -1, 'objective': -10}
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        foodList = self.getFood(successor).asList()
        myPos = successor.getAgentState(self.index).getPosition()

        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        defenders = [a for a in enemies if not a.isPacman and a.getPosition()
                     != None]
        inDanger = False
        if len(defenders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in defenders]
            # print(min(dists))
            inDanger = min(dists) <= 5

        # self.getScore(successor)
        features['successorScore'] = -len(foodList) if not inDanger else -21
        features['objective'] = 0 if not inDanger else self.getMazeDistance(
            myPos, self.futureObjective)
        # Compute distance to the nearest food
        if len(foodList) > 0:  # This should always be True,  but better safe than sorry
            minDistance = min([self.getMazeDistance(myPos, food)
                               for food in foodList])
            features['distanceToFood'] = minDistance if not inDanger else 100
        return features


    def getWeights(self, gameState, action):
        return {'successorScore': 100, 'distanceToFood': -1, 'objective': -10}

    #def saveWeights(self, gameState):



class DefensiveObjectiveAgent(ObjectiveCaptureAgent):
    """
    An objective agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """
    fileName = 'defensiveSave'
    weights = {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'objective': -0.5}

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman:
            features['onDefense'] = 0

        features['objective'] = self.getMazeDistance(
            myPos, self.futureObjective)

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i)
                   for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition()
                    != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(
                myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'objective': -0.5}
