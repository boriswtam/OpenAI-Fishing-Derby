#FishingDerby-v0

import gym
import time

def main():
    env = gym.make('FishingDerby-v0')
    env.reset()
  
    while True:
        env.render()
         # take a random action
        obs, rew, done, info = env.step(env.action_space.sample())
        time.sleep(0.02)
        if done:
            time.sleep(2)
            env.reset()

    # Image parsing alg information:
    # Fish:     color: 232, 232, 74     dim: 10 x 16 pix 
    #  102
    #  118
    #  134 (129-138)                
    #  150
    #  166
    #  182
    # Water color: 24, 26, 167
    # Top border: 0 - 6
    # Bottom border: 190 - 209
    # Borders (lft/rht):                dim: 8 pix
    # Pier                              dim: 20 pix (pole 4 pix) 0 - 27
    # Shark                             80 to 92 (13 x 25)

    # Ideas for what a state looks like:
    # (Where orient = to left/right, above/below)
    # (Fish ordered in distance from closest ot furthest away)
    #
    # No fish on hook & fi4(orient) & fi2(orient) & fi1(orient) & fi5(orient)
    #                   fi3(orient) & fi6(orient)
    # No fish on hook & fi4(orient) & fi2(orient)
    # Fish on hook & shark(orient)




if __name__ == '__main__':
    main()
