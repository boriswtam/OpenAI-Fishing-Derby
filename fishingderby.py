#FishingDerby-v0

import gym
import time

def main():
    env = gym.make('FishingDerby-v0')
    env.reset()
   
    # env.render()
    # obs, rew, done, info = env.step(env.action_space.sample()) # take a random action
    while True:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
        time.sleep(0.01)
        if done:
            time.sleep(2)
            env.reset()

    # 

    # Image parsing alg information:
    # Fish:     color: 232, 232, 74     dim: 10 x 16 pix 
    # Borders (lft/rht):                dim: 8 pix
    # Pier                              dim: 20 pix (pole 4 pix)
    # Shark                             80 to 92 (13 pix height)




if __name__ == '__main__':
    main()
