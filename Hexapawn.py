# create the board class to simulate hexapawn games


class Board():
    """
    Board instance contains a number of instance variables
        1. The board to store location of pieces, represented by a list with index of all squares and different ints to represent piece types
        2. The player whose turn it is to play, this breaks my intuition a bit because I expected that to be part of the game loop or something, neccesary if the board is
           to be able to generate its own legal moves.
        3. The Output index dictionary to store the corresponding output index of given moves in the Neural Network. 
           This seems to put the idea of the Neural Net into the structure of the the Board class,
           this in my opinion affects modularity so changing it to an external function is something to keep in mind
        4. Captures lists to help with move generation
        5. A legal moves cache to store the legal moves in a current position after generated
    Board instance also contains important methods
        1. method to set the starting position of the board by altering the board storage list
        2. method to get the indices of possible moves as that will serve as inputs to the Neural Network
        3. method to play a move on the board 
        4. method to generate legal moves, as I hinted at earlier, I'd probably have thought to have an external function determine the legal moves but this seems to be the convetion
        5. method for checking if it is a terminal position
        6. method for encoding the board position as a neural network input
        Interestinglg enough it lacks pop method to undo moves
    """
    EMPTY = 0 # different ints to represent white and black pieces and empty squares
    WHITE = 1
    BLACK = 2

    def __init__(self):
        self.turn = self.WHITE # game starts off with white to play

        self.WHITE_PAWN_CAPTURES = [ # list of possible capture destination for each square on the board, to help with the tricky legal moves generation
            [] ,
            [] ,
            [] ,
            [1] ,
            [0 ,2] ,
            [1] ,
            [4] ,
            [3 ,5] ,
            [4]
            ]
        self.BLACK_PAWN_CAPTURES = [
            [4] ,
            [3 ,5] ,
            [4] ,
            [7] ,
            [6 ,8] ,
            [7] ,
            [] ,
            [] ,
            []
            ]

        self.outputIndex = {} # cache for storing corresponding output for possible moves for easy conversion. Only stores possible white moves as will be explained later
        self.outputIndex["(6, 3)"] = 0
        self.outputIndex["(7, 4)"] = 1
        self.outputIndex["(8, 5)"] = 2
        self.outputIndex["(3, 0)"] = 3
        self.outputIndex["(4, 1)"] = 4
        self.outputIndex["(5, 2)"] = 5
        self.outputIndex["(6, 4)"] = 6
        self.outputIndex["(7, 3)"] = 7
        self.outputIndex["(7, 5)"] = 8
        self.outputIndex["(8, 4)"] = 9
        self.outputIndex["(3, 1)"] = 10
        self.outputIndex["(4, 0)"] = 11
        self.outputIndex["(4, 2)"] = 12
        self.outputIndex["(5, 1)"] = 13
        
        self.board = [self.EMPTY, self.EMPTY, self.EMPTY,
                      self.EMPTY, self.EMPTY, self.EMPTY,
                      self.EMPTY, self.EMPTY, self.EMPTY] # board is initialised as empty, could be initialised with starting position but that's left as an external method

    def setStartingPosition(self):
        self.board = [self.BLACK, self.BLACK, self.BLACK,
                      self.EMPTY, self.EMPTY, self.EMPTY,
                      self.WHITE, self.WHITE, self.WHITE] # arranges board to starting position, leaving it as a different method definetly 
                                                          # helps with flexibility like say rearranging the same Board object

    def getNetworkOutputIndex(self, move):
        """
        converts board moves to output index of the Neural Network. The Neural Network to be created only looks at positions from the perspective of the player to move,
        so all black moves can be mapped to the corresponding white move on a rotated board and then converted using the outputIndex cache
        """
        if move[0] > move[1]: # condition to check if it is a white move
            return self.outputIndex[str(move)]
        else:
            return self.outputIndex[str((8-move[0], 8-move[1]))] # subtracting a position from 8 inverts it on the storage list and rotates it on the board

    def applymove(self, move):
        """
        plays a specified move on the board. The legality of the move is not checked and wrongs move may throw an error or more likely and much worse make pieces do the impossible, 
        therefore it is assumed that anywhere the method is used, the moves were taken directly from generated legal moves.
        Moves are represented as tuples of starting piece position and ending piece position.
        Since pieces can't move backwards there is no ambiguity of which player the move applies to.
        """
        self.board[move[1]] = self.board[move[0]] # move whatever is on the starting square to the destination square. 
        self.board[move[0]] = self.EMPTY # convert the starting square to an empty square to show the piece has left
        if self.turn == self.WHITE: # change the whose turn it is to play
            self.turn = self.BLACK
        else:
            self.turn = self.WHITE
        self.legal_moves = None # reset the list of legal moves

    def generateMoves(self): 
        """
          The heart of the class, the bane of my existence for chess but for a simple game like Hexapawn, it is not that bad.
          The legal moves are generated by looping through all squares, checking if there's a piece on it, 
          generating all moves it could probably make and checking if they are indeed legal.
          This is very inefficient, if the storage was more sophisticated, it could probably the possible to loop through pieces not squares and other stuff,
          but then again this is Hexapawn and it is plenty efficient enough for it.
          Just like with the applymove method it runs the risk of doing illegal things, it assumes we are not in a terminal position and therefore must be used after a terminal check

          Including the terminal check into this method i.e. check if it is a terminal position and return an empty list if it is, is largely a function of taste

        """
        move = [] # create empty list to append moves to it as they are created
        for i in range(9): # for all squares on the board
            if self.board[i] == self.turn: # if the piece on said square is the piece to play, consider it else move along
                if self.turn == self.WHITE: # Do this for white
                    if (i-3) >= 0: # checks if the square in question has a square if front of it i.e. isn't  at the end of the bpoard
                        if self.board[i-3] == self.EMPTY: # if said square in front is empty...
                            move.append((i, i-3)) # ...append the pawn push
                        for capture in self.WHITE_PAWN_CAPTURES[i]: # for potentially possible capture destinations
                            if self.board[capture] == self.BLACK: # if an enemy pawn is there...
                                move.append((i, capture)) #...append the pawn capture

                if self.turn == self.BLACK: # essentially symmetrical to white
                    if (i+3) < 9:
                        if self.board[i+3] == self.EMPTY:
                            move.append((i, i+3))
                        for capture in self.BLACK_PAWN_CAPTURES[i]:
                            if self.board[capture] == self.WHITE:
                                move.append((i, capture))

        self.legal_moves = move
        return self.legal_moves 

    def isTerminal(self):
        """
        returns a tuple containing if the position is Terminal and the winner if it is
        """
        winner = None
        if (self.board[6] == self.BLACK or
            self.board[7] == self.BLACK or
            self.board[8] == self.BLACK): # pawn on the final rank for either player is a terminal position and a win for said player
            winner = self.BLACK
        if (self.board[0] == self.WHITE or
            self.board[1] == self.WHITE or
            self.board[2] == self.WHITE):
            winner = self.WHITE
        if winner != None: 
            return (True, winner)
        else:# if no terminal position thus far the only remaining terminal position is one with no moves, and the winner is whoever is not to move
            if len(self.generateMoves()) == 0:
                if self.turn == self.WHITE:
                    return (True, self.BLACK)
                else:
                    return (True, self.WHITE)
        return (False, None)
        # NB: I thought that the order of checking final rank wins before "stalemates" was important as stalemated person wins if she already has a pawn on the final rank,
        # but this obviously doesn't matter as a player cannot be in a stalemate position and have a final rank pawn at the same time as this will require their opponent playing after
        # they already reached the last rank

    def toNetworkInput(self):
        """
        converts the board position to an encoding that can be used in a Neural Network.
        In the book, the Neural Net was created to play for both sides so additional entries were required to specify whose turn it was.
        My Neural Net is designed to have the position pre transformed so it only ever sees it from the playing player's turn and it doesn't need these additional inputs.
        The book encoded the position by looping through the board and appending a one to the position vector list if the position had a white pawn and a zero if it didn't,
        essentially a one-hot encoding. The same thing is done for black and appended to the end of the list. 3 more entries are added to the list. 3 ones if it is white to move and 
        three zeros if it isn't. I do not quite understand why 3 bits were used instead of one but my guess is that it spreads the load among multiple inputs and allows it to work
        with smaller weights or a wider variety of weights leading to easier training. A similar wasteful approach was used in the AlphaZero paper that this entire project is inpired by
        To encode a position, I instead need to show it from the position of the player. To do this i always rotated the position if it was black to play by looping from the end.
        This removes the requirement for the last 3 bits and reduces the input to 18
        """
        posVec = []
        if self.turn == self.WHITE:
            for i in range(9):
                if self.board[i] == self.WHITE:
                    posVec.append(1)
                else:
                    posVec.append(0)
            for i in range(9):
                    if self.board[i] == self.BLACK:
                        posVec.append(1)
                    else:
                        posVec.append(0)
        else:
            for i in range(8, -1, -1): # starting from the end of the board effectively rotates the board
                if self.board[i] == self.BLACK:
                    posVec.append(1)
                else:
                    posVec.append(0)
            for i in range(8, -1, -1):
                if self.board[i] == self.WHITE:
                    posVec.append(1)
                else:
                    posVec.append(0)
        return posVec   
    # Remarks of what to adjust later

    # the idea of the Neural Network aiding feature being separated from the Board class to aid modularity

    # repition of code for white then black should be avoidable as this is a symmetric game so the same logic should be useable for both,
    # further more the theme of my approach as opposed to the original code in the book is symmetry so changing it feels natural.

    # quirk of this board class as opposed to many others is that it lacks a board.pop() method to reverse moves so it heavily relies on the copy module to fill the gap
    # it aids intuitive understanding especially for the minimax and MCTS but its unclear if it sacrifices efficiency. Unsure whether to change it or leave as is

    # there is no reverse method for converting output indices to corresponding moves on the board. Might have to change that eventually.

    # the legal_moves instance variable seems to have no real use so it could probably be removed although it could always be checked if it is non-empty in a
    # position that asked to generate moves before actually running the loop

    # the generatemoves method shouldn't have to loop through the entire  board as it is assumed that it isn't in a terminal position.
    # The range(9) could be conerveted to a range(6) but this should be done with care however as it requires the symmetery of black and white figured out to really be implemented  