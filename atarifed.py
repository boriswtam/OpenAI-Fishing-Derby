from keras.models import Sequential
from keras.layers import *
from collections import deque
from keras.models import model_from_json
import numpy as np
import gym
import random


env = gym.make('FishingDerby-v0')

numRepeats = 10
actions = 8
newModel = True
model = None
for i in range(numRepeats):
    if i > 0:
        print('Restarting process using same model as before, iteration %d' % i)
    else:
        if newModel:
            # environment state shape: (210, 160, 3)
            model = Sequential()
            model.add(Conv2D(32, kernel_size=(8, 8), strides= 4, activation='relu', input_shape=  env.observation_space.shape))
            model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
            model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(actions, init='uniform', activation='linear'))
            model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        else:
            model = load_model('modelSave.h5')

    print("input", env.observation_space.shape)
    for layer in model.layers:
        print("layer", layer.output_shape)

    D = deque()

    # Parameters to edit
    # number of time frame to observe
    observetime = 1500
    # Probability of doing a random move
    epsilon = 0.9
    # discount factor for future reward
    gamma = 0.5
    # learning batch size
    mb_size = 100

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
            action = np.random.randint(minAction, maxAction, size=1)[0]
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

        # Train network to output the Q function
        model.train_on_batch(inputs, targets)
        if i % 10 == 0:
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


    repNum = i // 10
    # serialize model to JSON
    modelSave = "savedModel" + str(repNum) + ".h5"
    modelFile = "model"+ str(repNum) + ".h5"
    json = "model" + str(repNum) + ".json"
    model.save(modelSave)
    model_json = model.to_json()
    with open(json, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelFile)
    print("Saved model to disk")

model.save("savedModel.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
