from keras.models import Sequential      # One layer after the other
from keras.layers import *  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from collections import deque            # For storing moves 
from keras.models import model_from_json
import numpy as np
import gym                                # To train our network
env = gym.make('FishingDerby-v0')         # Load fishing derby

import random     # For sampling batches from the observations

actions = 10

for i in range(10):
    if i > 0:
        print('Restarting process using same model as before, iteration %d' % i)
    else:
        # environment state shape: (210, 160, 3)
        # Create network. Input is two consecutive game states, output is Q-values of the possible moves.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(8, 8), strides= 4, activation='relu', input_shape=  env.observation_space.shape))
        #model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
        #model.add(MaxPooling2D(pool_size=(3, 3)))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(actions, init='uniform', activation='linear'))    # Same number of outputs as possible actions
        print("action space", actions)
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    print("input", env.observation_space.shape)
    for layer in model.layers:
        print("layer", layer.output_shape)

    # Pixels to info/objects to moves

    # Parameters
    D = deque()                                # Register where the actions will be stored

    observetime = 1500                 # Number of timesteps we will be acting on the game and observing results
    epsilon = 0.9                              # Probability of doing a random move
    #1: long term, 0: short term
    gamma = 0.5                              # Discounting factor for future reward. How much we care about steps further in time
    mb_size = 100                         # Learning minibatch size

    # Perform experience replay - by storing play data (observing)

    # FIRST STEP: Knowing what each action does (Observing)
    #get terms, frequency, get n gram; go through list and translate to freq; create new entry with document key
    observation = env.reset()                     # Game begins
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
    print("obs", obs.shape)
    state = np.stack((obs), axis=0)
    print("state", state.shape)
    done = False
    for t in range(observetime):
        #env.render()
        if np.random.rand() <= epsilon:
            action = np.random.randint(0, actions, size=1)[0]
            print("action", action)
        else:
            Q = model.predict(state)         # Q-values predictions
            #print("predictions", Q.shape)
            action = np.argmax(Q)             # Move with highest Q-value is the chosen one
            #print("action", action)

        observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
        #print("new obs", observation.shape)
        obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
        #print("new obs 2", obs_new.shape)
        #state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
        state_new = obs_new[:]
        #print("state new", state_new.shape)
        
        # D holds tuples of experience data
        D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
        state = state_new         # Update state
        if done:
            env.reset()           # Restart game if it's finished
            obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
            state = np.stack((obs), axis=0)
    print('Observing Finished')

    # SECOND STEP: Learning from the observations (Experience replay)


    minibatch = random.sample(D, mb_size)                              # Sample some moves
    # Minibatch is a tuple or experience data

    # initialize space
    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, actions))
    # targets = np.zeros((mb_size, env.action_space.n))


    for i in range(0, mb_size):
        print("learning", i)
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
        # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        
        # Approximate value function?
        Q_sa = model.predict(state_new)
        # Q_sa and targets contains numpy array(s) of predictions

        if done:
            targets[i, action] = reward
        else:
            # (Discounting factor & max prediction of Approx. val func) added to reward
            targets[i, action] = reward + gamma * np.max(Q_sa)

        # Train network to output the Q function
        # Inputs are the state (frame), and target data
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

    # Boris
    # Reuse model for more learning - name differently
    # Use model.save()
    # If program ends and you want to retrain the model on top of it
    #
    #
    # serialize model to JSON
    # model_json = model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # model.save_weights("model.h5")
    # print("Saved model to disk")

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
