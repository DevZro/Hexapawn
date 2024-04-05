from tensorflow import keras

# create a Neural Network model that will be later training to play Hexapawn

inp = keras.Input((18,)) # 18 inputs to match the 18 bit vector used to encode any position

l1 = keras.layers.Dense(64, activation="relu")(inp) # the book used 5 hidden layers of 128 nodes and stated it was overkill so hopefully this 2 layer of 64 will still be good enough
l2 = keras.layers.Dense(64, activation="relu")(l1)

"""
Neural Network to have 2 output heads, the policyHead to represent the output probability for all potentially possible moves, 
and the valueHead to represent the position evaluation.
"""
policyOut = keras.layers.Dense(14, activation="softmax", name="policyHead")(l2) # 14 outputs as expected and softmax because probabilities
valueOut = keras.layers.Dense(1, activation="tanh", name="valueHead")(l2) # tanh activation is used because evaluation should be 1 or -1

model = keras.Model(inp, [policyOut, valueOut])
model.compile(optimizer="SGD", loss={"policyHead" : keras.losses.CategoricalCrossentropy(),
                                      "valueHead" : "mean_squared_error"})
"""
mean_squared_error is popular with the valueOut while crossentropy is popular with the policyOut. The same choice was made for AlphaZero, 
may have to look into the reason in the future.
"""

model.save("random_model.keras")