import tensorflow as tf

import numpy as np

import constants as Constants
from master_network import MasterNetwork
from optimizer import Optimizer
from agent import Agent
from environment import Environment

import matplotlib.pyplot as plt

from keras.models import load_model

from saver import Saver
from savedata import SaveData

import gym, time, random, threading
           
####################
#  A3C CarRacing 
####################

# check action space
print ("Env: {}, Action Space: ".format(Constants.ENV))

data = SaveData()
# build master network
master_network = MasterNetwork(data)
summary = master_network.init_tf_summary()

#saver manages data and tf model saving
saver = Saver(data)

#testing env for playing in the end
env_test = Environment(gym.make(Constants.ENV), Agent(master_network, Constants.EPS_START, Constants.EPS_STOP, Constants.EPS_STEPS), summary, saver, render=True, eps_start=0., eps_end=0.)

#load last tf checkpoint
saver.load(master_network.session)

#dont train if replaymode is set
if not Constants.REPLAY_MODE:

    #initialize threads
    envs = [Environment(gym.make(Constants.ENV), Agent(master_network, Constants.EPS_START, Constants.EPS_STOP, Constants.EPS_STEPS), summary, saver) for i in range(Constants.THREADS-1)]
    #create cvshow thread
    envs.append(Environment(gym.make(Constants.ENV), Agent(master_network, Constants.EPS_START, Constants.EPS_STOP, Constants.EPS_STEPS), summary, saver, render=True, cvshow=True))

    opts = [Optimizer(master_network) for i in range(Constants.OPTIMIZERS)]

    #start time
    saver.data.start_time = time.time() - saver.data.wall_t

    # start threads
    for o in opts:
        o.start()
    for e in envs:
        e.start()

    while data.global_t<Constants.RUN_TIME:
        time.sleep(10)

    for e in envs:
        e.stop()
    for e in envs:
        e.join()
        
    for o in opts:
        o.stop()
    for o in opts:
        o.join()
    
master_network.init_tf_summary()

######## plot frames datas ########
plt.plot(data.frames_t,data.v)
plt.show()
###################################

print("Training finished")
saver.save(master_network.session)

env_test.run()