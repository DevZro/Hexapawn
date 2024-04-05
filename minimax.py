import copy
from Hexapawn import Board

# create minimax function to be used to generate training data for the Neural Network
# NB: alpha-beta pruning isn't using because it cuts out positions and the Network ends up never seeing them

def minimax(board, input_list, output_list1, output_list2):
    """
    Standard 2 output minimax function. It is modified to generate the evaluation for every possible Hexapawn position and append them to a list.
    I considered making it a pure minimax function and leaving the process of extracting training positions out to another function for the sake of modularity,
    but using a tree traversal like this has a time complexity of O(n) while generating all possible positions and using minimax on them individually has a complexity
    of O(nlogn). 
    """

    if board.isTerminal()[0]:
        if board.isTerminal()[1] == Board.WHITE:
            return 1, None
        else:
            return -1, None
        
    if board.turn == Board.WHITE:
        best_score = -10
        best_move = None
        for move in board.generateMoves():
            tmp = copy.deepcopy(board)
            tmp.applymove(move)
            score = minimax(tmp, input_list, output_list1, output_list2)[0]
            if score > best_score:
                best_score = score
                best_move = move
    
    else:
        best_score = 10
        best_move = None
        for move in board.generateMoves():
            tmp = copy.deepcopy(board)
            tmp.applymove(move)
            score = minimax(tmp, input_list, output_list1, output_list2)[0]
            if score < best_score:
                best_score = score
                best_move = move

    input_list.append(board.toNetworkInput()) # convert each non-terminal position into NN input and append them to the data_list
    move_vector = [0 for i in range(14)]
    move_vector[board.getNetworkOutputIndex(best_move)] = 1 # change only the index of the best_move to 1 and leave the rest as 0s
    output_list1.append(move_vector)
    output_list2.append(best_score if board.turn == Board.WHITE else - best_score) # the NN sees positions from player's perspective so scores should be adjusted to show that

    return best_score, best_move


# Remark

# The minimax function should be changed to a Negamax as per my symmetry theme. This however will be saved for the next commit