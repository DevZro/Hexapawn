import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Hexapawn import Board
from minimax import minimax
import json
import random
# from mcts import ReinfLearn
from model import HexaPawnNet

# Part 1
# training a NN using supervised Learning method
model = HexaPawnNet()

board = Board()
x = []
target_policy = []
target_value = []
mask_list = []


minimax(board, x, target_policy, target_value, mask_list) # generate training data

data = {
    "x": x,
    "target_policy": target_policy,
    "target_value": target_value,
    "mask_list": mask_list
}

with open("data.json", "w") as file:
    json.dump(data, file, indent=4)

x = torch.tensor(x, dtype=torch.float32)
target_policy = torch.tensor(target_policy, dtype=torch.float32)
target_value = torch.tensor(target_value, dtype=torch.float32)

mask = torch.zeros((len(mask_list), 14), dtype=torch.bool)

for i, idxs in enumerate(mask_list):
    mask[i, idxs] = True

class HexaPawnDataset(Dataset):
    def __init__(self, states: torch.Tensor, policies: torch.Tensor, values: torch.Tensor,  masks: torch.Tensor):
        self.states = states.float()
        self.masks = masks
        self.policies = policies.float()  
        self.values = values.float()  
        self.len = self.states.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx], self.masks[idx]

def loss_fn(policy_logits, value, target_policy, target_value, move_mask): # loss function
    value_loss = F.mse_loss(value, target_value)

    policy_logits = policy_logits.masked_fill(~move_mask, -1e9) # mask illegal moves
    policy_loss = F.cross_entropy(policy_logits, target_policy.argmax(dim=-1))
    
    return policy_loss + value_loss

def train_step(model, optimizer, x, target_policy, target_value, move_mask):
    optimizer.zero_grad()
    policy_logits, value = model(x)
    loss = loss_fn(policy_logits, value, target_policy, target_value, move_mask)
    loss.backward()
    optimizer.step()
    return loss.item()

batch_size = 16
dataset = HexaPawnDataset(x, target_policy, target_value, mask)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
model.train()
for epoch in range(512):  # train for 512 epochs
    total_loss = 0
    for batch_states, batch_policies, batch_values, batch_masks in dataloader:
        loss = train_step(model, optimizer, batch_states, batch_policies, batch_values, batch_masks)
        total_loss += loss
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")

torch.save(model.state_dict(), "supervised_model.pth")


def rand_vs_network(model, use_mask=True):
    
    #function to simulate match between a random player as the white pieces and the NN as the black pieces.
    
    board = Board()

    while  not board.isTerminal()[0]: # loop for each move i.e. 1 ply for each player
        move = random.choice(board.generateMoves())
        board.applyMove(move)

        if board.isTerminal()[0]: # check for a win after the random player plays
            break
        else:
            
            #the method of playing a move by the NN is clunky and may have to be adjusted in a later commit
           
            network_output = model(torch.tensor([board.toNetworkInput()], dtype=torch.float32))[0][0]  # get the policy output for the current position

            # There shouldn't be a need to use move masks if the model is trained normally
            # but it was trained with a mask so it didn't learn to not suggest illegal moves
            if use_mask:
                mask = torch.zeros((14,), dtype=torch.bool)
                idx = [board.getNetworkOutputIndex(move) for move in board.generateMoves()]
                mask[idx] = True

                # replaces illegal moves logits with a very large negative number
                # this removes all useless values
                network_output = network_output.masked_fill(~mask, -1e9)
                network_output = F.softmax(network_output, dim=0)

            move_index = torch.argmax(network_output).item() # find the move index of the NN's choice
            best_move = None
            for move in board.generateMoves():
                
                #loop through all the legal moves to find out which one has the index of the NN's choice. 
                #The clunkiness could be fixed by a dictionary and probably will be.
                
                if board.getNetworkOutputIndex(move) == move_index:
                    best_move = move
                    break
            board.applyMove(best_move)

    return board.isTerminal()[1] # returns winner

white_win = 0
black_win = 0

for i in range(1000): # quick round of 100 games to see if the NN is indeed perfect
    result = rand_vs_network(model)
    if result == Board.WHITE:
        white_win += 1
    elif result == Board.BLACK:
        black_win += 1
    if (i % 100) == 99:
        print(f"Game {i + 1} complete!")
print(f"Out of a 1000 games, the random player won {white_win} while the Neural Net won {black_win}")

"""
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
"""