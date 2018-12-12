from keras.models import Sequential
from keras.layers import *
from collections import deque
from keras.models import model_from_json
import numpy as np
import gym
import random

env = gym.make('FishingDerby-v0')
#env: (210, 160, 3)
model = Sequential()


actions = 8
model.add(Conv2D(32, kernel_size=(8, 8), strides= 4, activation='relu', input_shape=  env.observation_space.shape))
model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(actions, init='uniform', activation='linear'))
print("action space", actions)
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

print("input", env.observation_space.shape)
for layer in model.layers:
    print("layer", layer.output_shape)

D = deque()
# Parameters to edit
#time spent observing
observetime = 10000
# Probability of doging a random move
epsilon = 0.99
# discounting factor
gamma = 0.55
# learning batch size
mb_size = 200

# FIRST STEP: Knowing what each action does (Observing)
observation = env.reset()
obs = np.expand_dims(observation, axis=0)
print("obs", obs.shape)
state = np.stack((obs), axis=0)
print("state", state.shape)
done = False
minAction = 2
maxAction = 10
for t in range(observetime):
    #env.render()
    if np.random.rand() <= epsilon:
        action = np.random.randint(0, actions, size=1)[0]
    else:
        Q = model.predict(state)
        action = np.argmax(Q)

    observation_new, reward, done, info = env.step(action)
    obs_new = np.expand_dims(observation_new, axis=0)
    state_new = obs_new[:]

    D.append((state, action, reward, state_new, done))
    state = state_new
    if done:
        env.reset()
        obs = np.expand_dims(observation, axis=0)
        state = np.stack((obs), axis=0)
print('Observing Finished')

# SECOND STEP: Learning from the observations (Experience replay)
minibatch = random.sample(list(D), mb_size)
inputs_shape = (mb_size,) + state.shape[1:]
inputs = np.zeros(inputs_shape)
targets = np.zeros((mb_size, actions))

for i in range(0, mb_size):
    print("learning", i)
    state = minibatch[i][0]
    action = minibatch[i][1] - minAction
    reward = minibatch[i][2]
    state_new = minibatch[i][3]
    done = minibatch[i][4]

    # Build Bellman equation for the Q function
    inputs[i:i+1] = np.expand_dims(state, axis=0)
    targets[i] = model.predict(state)
    Q_sa = model.predict(state_new)

    if done:
        targets[i, action] = reward
    else:
        targets[i, action] = reward + gamma * np.max(Q_sa)

    model.train_on_batch(inputs, targets)
    if i % 100 == 0:
        # serialize model to JSON
        model_json = model.to_json()
        file = "model" + str(i) + ".json"
        with open(file, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        file_name = "model" + str(i) + ".h5"
        print("saving model to ", file_name, file)
        model.save_weights(file_name)
print('Learning Finished')


# serialize model to JSON
model.save("savedModel.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
