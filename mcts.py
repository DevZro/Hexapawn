import copy
import math
import random
import numpy as np
from Hexapawn import Board

# create the Monte-Carlo Tree Search to be used an alternative to generate training data and also as a facility for Reinforcement Learning
# This is my first encounter with MCTS so admittedly a lot of my ideas are derivative from the book

class Edge:
    """
    A class for edges connecting Nodes. Serves as an object that sits on specific moves and holds all the variables used for the MCTS which I'd expect from my little experience
    with data structures to be held by the Node. Also the MCTS doesn't seem to be a tree but much closer to a graph but oh well.
    """
    def __init__(self, move, parentNode):
        """
        initialised with the specific move and the parentNode it is connected to, this also breaks my intuition because linked-list style structures generally either have both
        parent and child or only child. This is not a problem though because the parentNode is more conventional and stores not only the childEdge but also childNode.
        Stores 4 additional instance variables
        1. N to represent the number of times the move has been played during a game
        2. P to represent the network prior-probability of the move (as in the probability the network assigns to the move without being guided by MCTS)
        3. W to represent the sum total of evaluations of games occuring from that move from the perspective of the player. 
        4. Q to represent the mean evaluation of the position
        """
        self.move = move 
        self.parentNode = parentNode
        self.N = 0
        self.P = 0
        self.Q = 0 # detail that was barely mentioned in the book but self.Q is initialised to 0 because the extreme evualuations are -1 and 1 not 0 and 1
        self.W = 0

class Node:
    """
    initialised with the specific board position just like the nodes of a minimax tree but also with a parentEdge
    """

    def __init__(self, board, parentEdge):
        self.board = board
        self.parentEdge = parentEdge
        self.childEdgeNode = []

    def expand(self, network):
        """
        Expand a leaf position to add all possible resulting position to the tree and initialise edges with the priror network probabilities
        """
        q = network.predict(np.array([self.board.toNetworkInput()]))
        total_prob = 0
        for move in self.board.generateMoves():
            childEdge = Edge(move, self)
            childEdge.P = q[0][0][self.board.getNetworkOutputIndex(move)]
            total_prob += childEdge.P
            tmp = copy.deepcopy(self.board) # just like with the minimax, a new board object is created for every new added node
            tmp.applymove(move)
            childNode = Node(tmp, childEdge)
            self.childEdgeNode.append((childEdge, childNode))

        for (edge, _) in self.childEdgeNode:
            edge.P /= total_prob # scale up the probabilities of legal moves after removing illegal moves
        
        return q[1][0][0] # returns predicted network evaluation, will be useful during the expand_and_evaluate step
    
    def isLeaf(self):
        return len(self.childEdgeNode) == 0
    
class MCTS:

    def __init__(self, network):
        self.network = network
        self.rootNode = None
        self.c_puct = 1
        self.tau = 1

    def uct_value(self, edge, parentN):
        """
        Good old fashioned UCT, slightly altered from the textbook version to accomodate the use of a NN
        """
        return self.c_puct * edge.P * (math.sqrt(parentN))/(edge.N + 1)
    
    def select(self, node):
        walk = node
        while not walk.isLeaf():
            best_nodes= [] # incase there are multiple equally good nodes
            max_uct_value = -1000000
            for (edge, node) in walk.childEdgeNode: # find maximum "uct_value"
                uct_value = edge.Q + self.uct_value(edge, walk.parentEdge.N)
                if uct_value > max_uct_value:
                    max_uct_value = uct_value
            for (edge, node) in walk.childEdgeNode: # append all equally good nodes and pick a random one
                if  edge.Q + self.uct_value(edge, walk.parentEdge.N) == max_uct_value:
                    best_nodes.append(node)
            walk = random.choice(best_nodes) # choosing at random helps with exploration of the tree
        return walk
    
    def expand_and_evaluate(self, node):
        """
          attempt to show the evaluation from the perspective of the player, the weird negation is a side-effect of the applymove method changing the player turn immediately
          after a move is made. Therefore in a terminal position, the last player won't be shown when board.turn is checked.  
        """
        if node.board.isTerminal()[0]:
            v = -1 if node.board.isTerminal()[1] == node.board.turn else 1 
            self.backpropagate(v, node.parentEdge)
        else:
            v = - node.expand(self.network) 
            self.backpropagate(v, node.parentEdge)

    def backpropagate(self, v, edge):
        edge.N += 1
        edge.W += v
        edge.Q = edge.W/edge.N
        if edge.parentNode:
            if edge.parentNode.parentEdge:
                self.backpropagate(-v, edge.parentNode.parentEdge) # -v because a win for a player is a loss for a player one up the tree and vice-versa

    def search(self, rootNode):
        self.rootNode = rootNode
        self.rootNode.expand(self.network) 
        for _ in range(100): # play a sequence of moves 100 times to build up statistical information
            node = self.select(rootNode)
            self.expand_and_evaluate(node)
        move_prob = []
        for (edge, node) in self.rootNode.childEdgeNode:
            prob = (edge.N ** (1/self.tau))/(sum(edge.N **(1/self.tau) for (edge, _) in self.rootNode.childEdgeNode)) # find the probability of each move using Node count
            move_prob.append((edge.move, prob, edge.N, edge.Q))
        return move_prob

class ReinfLearn:

    def __init__(self, model):
        self.model = model

    def playGame(self):
        """
        play a game using the same network and MCTS and record all positions that occurred in the games alongside the move_prob and the eventual score of each player
        """
        positionData = []
        moveProbData = []
        valueData = []

        board = Board()
        board.setStartingPosition()

        while not board.isTerminal()[0]:
            positionData.append(board.toNetworkInput())
            rootEdge = Edge(None, None)
            rootNode = Node(board, rootEdge)
            rootEdge.N = 1 # the rootEdge is give a N of 1 else uct of the children edges throw errors
            mcts = MCTS(self.model)
            moveVector = [0 for _ in range(14)]
            moveProb = mcts.search(rootNode) # use MCTS
           
            for (move, prob, _ , _) in moveProb:
                moveVector[board.getNetworkOutputIndex(move)] = prob # records the probability of each move index

            rand_idx = np.random.multinomial(1, moveVector) # use numpy to choose a random move_index based on the probabilities
            idx = np.where(rand_idx==1)[0][0]
            move_choice = None

            for (move, prob, _, _) in moveProb: # find said move
                if board.getNetworkOutputIndex(move) == idx:
                    move_choice = move
            
            moveProbData.append(moveVector)
            board.applymove(move_choice)

        if board.isTerminal()[1] == board.WHITE:
            for i in range(len(positionData)): # if white won that every black position has a score of -1
                valueData.append((-1)**i)

        else:
            for i in range(len(positionData)): # opposite
                valueData.append((-1)**(i+1))
            
        return (positionData, moveProbData, valueData)


            
# Remark

# May change the Edge class to be more like a traditional linked list with a parent Node and child Node, that way the Node only needs to have childEdges instead of childEdgeNodes

# the uct_value method does not need to have the parentN argument. Should change that in a later commit






    




            