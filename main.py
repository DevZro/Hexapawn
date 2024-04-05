from tensorflow import keras
from Hexapawn import Board
from minimax import minimax
import numpy as np
import random
from mcts import ReinfLearn

# Part 1
# training a NN using supervised Learning method
model = keras.models.load_model("random_model.keras") # load untrained model

board = Board()
board.setStartingPosition()

inputData = []
moveProbData = []
valueData = []

minimax(board, inputData, moveProbData, valueData) # generate training data

inputData = np.array(inputData)
moveProbData = np.array(moveProbData)
valueData = np.array(valueData)

np.save("inputData", inputData)
np.save("moveProbData", moveProbData)
np.save("valueData", valueData)

model.fit(inputData, [moveProbData, valueData], epochs=512, batch_size=16) # train model, 512 epochs may actually be overkill
model.save("supervised_model.keras")

def rand_vs_network(model):
    
    #function to simulate match between a random player as the white pieces and the NN as the black pieces.
    
    board = Board()
    board.setStartingPosition()

    while  not board.isTerminal()[0]: # loop for each move i.e. 1 ply for each player
        move = random.choice(board.generateMoves())
        board.applymove(move)

        if board.isTerminal()[0]: # check for a win after the random player plays
            break
        else:
            
            #the method of playing a move by the NN is clunky and may have to be adjusted in a later commit
           
            network_output = model.predict(np.array([board.toNetworkInput()]))[0][0]  # get the policy output for the current position
            move_vector = [0 for i in range(14)]
            for move in board.generateMoves(): # sets illegal move predictions by the NN to 0
                move_vector[board.getNetworkOutputIndex(move)] = network_output[board.getNetworkOutputIndex(move)]
            move_index = np.argmax(np.array(move_vector)) # find the move index of the NN's choice
            best_move = None
            for move in board.generateMoves():
                
                #loop through all the legal moves to find out which one has the index of the NN's choice. 
                #The clunkiness could be fixed by a dictionary and probably will be.
                
                if board.getNetworkOutputIndex(move) == move_index:
                    best_move = move
            board.applymove(best_move)

    return board.isTerminal()[1] # returns winner

white_win = 0
black_win = 0

for i in range(100): # quick round of 100 games to see if the NN is indeed perfect
    if rand_vs_network(model) == Board.WHITE:
        white_win += 1
    else:
        black_win += 1

print(f"Out of a 100 games, the random player won {white_win} while the Neural Net won {black_win}")

#Remark

# It is important to note that the Neural Network is used to show a concept therefore it is trained to overfit and essentially memorise every Hexapawn position.
# This is quite different from what will be desired for an actual use like chess but for a toy game like Hexapawn, it's fine.


# Part 2
# training a NN using Reinforcement learning

model = keras.models.load_model("random_model.keras") # load random_model
learner = ReinfLearn(model)

for i in range(10): # use reinforcement learning 10 times to improve the model
    inputData = []
    outputData1 = []
    outputData2 = []
    for j in range(20): # play 20 games and add all their positions to creat training data
        data = learner.playGame()
        inputData += data[0]
        outputData1 += data[1]
        outputData2 += data[2]
    model.fit(np.array(inputData), [np.array(outputData1), np.array(outputData2)], epochs=512, batch_size=16) # 512 epochs might again be an overkill
    model.save(f"reinforced_model{i}.keras") # save current iteration of Reinforced model

score = []
for i in range(10): # test all 10 saved iterations of the reinforced model
    model = keras.models.load_model(f"reinforced_model{i}.keras")
    white_win = 0
    black_win = 0
    for i in range(100): # quick round of 100 games to see if the NN is indeed perfect
        if rand_vs_network(model) == Board.WHITE:
            white_win += 1
        else:
            black_win += 1
    score.append(black_win)
for black_win in score:
    print(f"Out of a 100 games, the random player won {(100 - black_win)} while the Neural Net won {black_win}")
