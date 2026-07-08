# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()
        food = newFood.asList()
        minDistance = float('inf')
        for f in food:
            d = util.manhattanDistance(newPos, f)
            if d < minDistance:
                minDistance = d
        score += 1.0 / minDistance
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            scared = ghostState.scaredTimer
            d = util.manhattanDistance(ghostPos, newPos)
            if scared == 0:
                if d <= 2:
                    return -float('inf')
                elif d <= 4:
                    score -= 4.0 / d
                else:
                    score -= 3.0 / d
            else:
                if d > 0:
                    score += 2.0 / d
        return score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth += 1 
        if agentIndex == 0:
            maxEval = -float('inf')
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                eval = self.minimax(successor, nextAgent, nextDepth)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            minEval = float('inf')
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                eval = self.minimax(successor, nextAgent, nextDepth)
                minEval = min(minEval, eval)
            return minEval


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestValue = -float('inf')
        action = None
        legalMoves = gameState.getLegalActions(0)
        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            eval = self.minimax(successor, 1, 0)
            if eval > bestValue:
                bestValue = eval
                action = move
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabeta(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth += 1 
        if agentIndex == 0:
            maxEval = -float('inf')
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                eval = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
                maxEval = max(maxEval, eval)
                alpha = max(alpha, eval)
                if beta < alpha:
                    break
            return maxEval
        else:
            minEval = float('inf')
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                eval = self.alphabeta(successor, nextAgent, nextDepth, alpha, beta)
                minEval = min(minEval, eval)
                beta = min(beta, eval)
                if beta < alpha:
                    break
            return minEval

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -float('inf')
        beta = float('inf')
        bestValue = -float('inf')
        action = None
        legalMoves = gameState.getLegalActions(0)
        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            eval = self.alphabeta(successor, 1, 0, alpha, beta)
            if eval > bestValue:
                bestValue = eval
                action = move
            alpha = max(alpha, eval)
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        nextDepth = depth
        if nextAgent == 0:
            nextDepth += 1 
        if agentIndex == 0:
            maxEval = -float('inf')
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                eval = self.expectimax(successor, nextAgent, nextDepth)
                maxEval = max(maxEval, eval)
            return maxEval
        else:
            total = 0
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                eval = self.expectimax(successor, nextAgent, nextDepth)
                total += eval
            return total / len(legalMoves)

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestValue = -float('inf')
        action = None
        legalMoves = gameState.getLegalActions(0)
        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            eval = self.expectimax(successor, 1, 0)
            if eval > bestValue:
                bestValue = eval
                action = move
        return action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    capsules = currentGameState.getCapsules()
    score = currentGameState.getScore()
    result = score
    if food:
        d = min(util.manhattanDistance(pos, f) for f in food)
        result += 1.0 / d
    result -= 4 * len(food)
    result -= 10 * len(capsules)
    for g in ghostStates:
        distance = util.manhattanDistance(pos, g.getPosition())
        if g.scaredTimer > 0:
            result += 20.0 / distance
        else:
            if distance <= 2:
                result -= float('inf')
            else :
                result -= 3.0 / distance
    return result

# Abbreviation
better = betterEvaluationFunction
