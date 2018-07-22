import multiprocessing, time, random, threading
from multiprocessing import Process, Pipe, Queue
import numpy as np
import tensorflow as tf
from osim.env import ProstheticsEnv


class penv(ProstheticsEnv):
    """Base class of altered environments
    
    Every type of environment trained in the challenge should be inhereted
    from this class. Wether the reward function is different or we use
    a different type of observation, etc...
    """

    def __init__(self, visualize):
        ProstheticsEnv.__init__(self, visualize=visualize)


if __name__ == '__main__':
    env = penv(visualize=False)
    observation = env.reset()
    for i in range(200):
        print(i)
        observation, reward, done, info = env.step(env.action_space.sample())
