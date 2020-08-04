 # plots.py
 # author: aziz jegham
 # Created on Tue Aug 04 2020
 # Copyright (C) 2020 aziz jegham
 # License: GNU General Public License version 3

from ..simulator import Simulator
from ..utils.utils import rollingAverage
from ..utils.utils import progressBar

import sys
import numpy as np
from matplotlib import pyplot as plt

def plotRewards(simulator_p, nsteps_p):
    for step_cntr in range(nsteps_p):
        simulator_p.nextStep()
    
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("reward per step")
    plt.scatter(range(1, nsteps_p+1), simulator_p.getRewardsList())
    plt.show()

def plotEvals(simulator_p, nsteps_p):
    evals_l = []
    for step_cntr in range(nsteps_p):
        evals_l.append(simulator_p.policy_.vectBanditsEvals_.copy())
        simulator_p.nextStep()
    
    plt.xlabel("step")
    plt.ylabel("evaluation")
    plt.title("bandits evaluations evolution")
    for bandit_cntr in range(1, 1+len(evals_l[0])):
        bandit_evals = [stepEvals_l[bandit_cntr-1] for stepEvals_l in evals_l]
        plt.plot(bandit_evals, label='bandit %s'%bandit_cntr)
    plt.legend()
    plt.show()

def plotAggregates(simulator_p, nsteps_p, runs_p, window_p = 10):
    # Simulate runs and store results
    evals_l = [] # eval[run][step][bandit]
    rewards_l = [] # reward[run][step]
    for run_cntr in range(runs_p):
        progressBar(run_cntr, runs_p)
        runEvals_l = []
        for step_cntr in range(nsteps_p):
            runEvals_l.append(simulator_p.policy_.vectBanditsEvals_.copy())
            simulator_p.nextStep()
        evals_l.append(runEvals_l)
        rewards_l.append(simulator_p.getRewardsList())
        simulator_p.reinit()
    
    # Aggregate evaluations 
    evalsArray_l  = np.array(evals_l)
    aggEvals_l = []
    for step_cntr in range(nsteps_p):
        aggStepEvals_l = []
        for bandit_cntr in range(evalsArray_l.shape[2]):
            aggStepEvals_l.append(np.mean(evalsArray_l[:, step_cntr, bandit_cntr]))
        aggEvals_l.append(aggStepEvals_l) 
    
    # Plot aggregate evaluations 
    plt.figure()
    plt.xlabel("step")
    plt.ylabel("evaluation")
    plt.title("bandits aggregate evaluations evolution (%s runs)"%runs_p)
    for bandit_cntr in range(1, 1+len(aggEvals_l[0])):
        banditEvals_l = [stepEvals_l[bandit_cntr-1] for stepEvals_l in aggEvals_l]
        plt.plot(banditEvals_l, label='bandit %s'%bandit_cntr)
    plt.legend()

    # Aggregate rewards 
    rewardsArray_l = np.array(rewards_l)
    aggRewards_l = []
    for step_cntr in range(nsteps_p):
        aggRewards_l.append(np.mean(rewardsArray_l[:, step_cntr]))
    
    # Rolling average rewards 
    rollingRewards_l = rollingAverage(aggRewards_l, window_p)
    
    # Plot aggregate rewards 
    plt.figure()
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("aggregate rewards evolution (%s runs)"%runs_p)
    plt.plot(range(1, nsteps_p+1), aggRewards_l, label="reward per step")
    plt.plot(range(1, nsteps_p+1), rollingRewards_l, "r", label='mean reward over last %i steps'%window_p)
    plt.legend()

    plt.show()