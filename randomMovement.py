import numpy as np
import gym                                # To train our network
env = gym.make('FishingDerby-v0')          # Choose game (any in the gym should work)

import random     # For sampling batches from the observations

done = False
tot_reward = 0.0
env.reset()
while not done:
    env.render()                    # Uncomment to see game running
    action = np.random.randint(2, 10, size=1)[0]
    observation, reward, done, info = env.step(action)
    #obs = np.expand_dims(observation, axis=0)
    #print("new state", state.shape)
    tot_reward += reward
print('Game ended! Total reward: {}'.format(tot_reward))
