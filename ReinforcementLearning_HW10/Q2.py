import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":



    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            s = obs
            action = None
            if random.uniform(0,1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)])
                action =  np.argmax(prediction)

            a = action
            
            obs,reward,done,info = env.step(action)
            s2 = obs
            r = reward
            episode_reward += r
            if done:
               episode_reward_record.append(episode_reward)

            best_reward = max([Q_table[(s2,i)] for i in range(env.action_space.n)])
            if not done:
                Q_table[(s,a)] += LEARNING_RATE *(r + (DISCOUNT_FACTOR*(best_reward)) - Q_table[(s,a)]) 
            else:
                Q_table[(s,a)] += LEARNING_RATE *(r - Q_table[(s,a)]) 
        EPSILON = EPSILON * EPSILON_DECAY
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )
    
    
    model_file = open('Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    model_file.close()








