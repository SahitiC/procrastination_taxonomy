"""
define quantities that are held constant across scripts
"""

import numpy as np

# define some standard params:
# states of markov chain
STATES_NO = 22+1  # one extra state for completing nothing
STATES = np.arange(STATES_NO)

# actions = no. of units to complete in each state
# allow as many units as possible based on state
ACTIONS = [np.arange(STATES_NO-i) for i in range(STATES_NO)]

HORIZON = 16  # no. of weeks for task
DISCOUNT_FACTOR = 0.9  # discounting factor
EFFICACY = 0.8  # self-efficacy (probability of progress for each unit)

# utilities :
REWARD_THR = 4.0  # reward per unit at threshold (14 units)
REWARD_THR_DIFF_DISCOUNTS = 1
REWARD_EXTRA = REWARD_THR/4  # reward per unit post-threshold upto 22 units
REWARD_EXTRA_DIFF_DISCOUNTS = REWARD_THR_DIFF_DISCOUNTS/4
REWARD_SHIRK = 0.0
EFFORT_WORK = -0.3
BETA = 5
BETA_DIFF_DISCOUNTS = 10

THR = 14  # threshold number of units for rewards
N_TRIALS = 20  # no. of trajectories per dataeset for recovery
# N = 1000  # no of params sets to recover

# params for no commitment model
STATES_NO_NO_COMMIT = (22+1) * 2
STATES_NO_COMMIT = np.arange((22+1) * 2)
ACTIONS_BASE = [np.arange(23-i) for i in range(23)]
INTEREST_STATES = np.array([0, 1])
P_STAY_LOW = 0.95
P_STAY_HIGH = 0.05
REWARD_INTEREST = 2.0
