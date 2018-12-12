from keras.models import Sequential
from keras.layers import Dense, Flatten
from collections import deque
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
import gym
import random

env = gym.make('FishingDerby-v0')
# load json and create model
json_file = open('goodmodel.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# load weights into new model
model.load_weights("goodmodel.h5")

#model = load_model('modelSave.h5')
#print("Loaded model from disk")

print("\nenv", env.observation_space.shape)
print(env.action_space.n)
print(env.action_space.shape)
observation = env.reset()
print(observation.shape)
state = np.expand_dims(observation, axis=0)
print("new state", state.shape)
done = False
tot_reward = 0.0
while not done:
    env.render()
    Q = model.predict(state)
    action = np.argmax(Q)
    observation, reward, done, info = env.step(action)
    state = np.expand_dims(observation, axis=0)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(tot_reward))
