import multiprocessing, time, random, threading
from multiprocessing import Process, Pipe, Queue
import numpy as np
import tensorflow as tf
from osim.env import ProstheticsEnv


class penv(object):
    pass


if __name__ == '__main__':
    env = ProstheticsEnv(visualize=False)
    observation = env.reset()
    for i in range(200):
        observation, reward, done, info = env.step(env.action_space.sample())
