"""
Functions for constructing the reward functions and transition matrices for
Zhang and Ma (2023) NYU study
"""
import numpy as np
# from scipy.stats import binom
from scipy.special import comb


def reward_threshold(states, actions, reward_shirk, reward_thr,
                     reward_extra, thr, states_no):
    """
    construct reward function where units are rewarded immediately once
    threshold no. of units are hit (compensated at reward_thr per unit) & then
    reward_extra per every extra unit until max no. of states units
    (in Zhang and Ma data, thr=14 and max no of units = 22); reward for
    shirking is immediate

    params:
        states (ndarray): states of an MDP
        actions (list): actions available in each state
        reward_shirk (float): reward for doing an alternate task (i.e., for
        each unit that is not used for work)
        reward_thr (float): reward for each unit of work completed until thr
        reward_extra (float): reward for each unit completed beyond thr
        thr (int): threshold number of units until which no reward is obtained
        states_no (int): max. no of units that can be completed

    returns:
        reward_func (list): rewards at each time point on taking each action at
        each state

    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # if less than thr credits have been completed, then thresholded reward
        # at thr units, flatter extra rewards till states_no and then flat
        if state_current < thr:

            for i, action in enumerate(
                    actions[state_current][thr-state_current:
                                           states_no-state_current]):

                reward_temp[action, thr:action+state_current+1] += (
                    thr*reward_thr
                    + np.arange(0, action+state_current+1-thr, step=1)
                    * reward_extra)

            for i, action in enumerate(actions[state_current]
                                       [states_no-state_current:]):

                reward_temp[action, thr:states_no] += np.arange(
                    thr*reward_thr,
                    thr*reward_thr + (states_no-thr)*reward_extra,
                    step=reward_extra)
                reward_temp[action, states_no:action+state_current+1] += (
                    thr*reward_thr + (states_no-1-thr)*reward_extra)

        # if more than 14 units completed, extra reward unitl 22 is reached
        # and then nothing
        elif state_current >= thr and state_current < states_no-1:

            for i, action in enumerate(actions[state_current]
                                       [:states_no-state_current]):

                reward_temp[action, state_current+1:
                            action+state_current+1] += (
                                np.arange(1, action+1)*reward_extra)

            # reward_temp[states_no-state_current:, :] = reward_temp[
            # states_no-1-state_current, :]
        reward_func.append(reward_temp)

    return reward_func


def reward_immediate(states, actions, reward_shirk,
                     reward_unit, reward_extra):

    reward_func = []

    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        # immediate rewards for units completed
        for action in range(len(actions[state_current])):

            reward_temp[action, state_current:state_current+action+1] += (
                np.arange(0, action+1) * reward_unit
            )

        reward_func.append(reward_temp)

    return reward_func


def reward_no_immediate(states, actions, reward_shirk):
    """
    The only immediate rewards are from shirk
    """

    reward_func = []
    for state_current in range(len(states)):

        reward_temp = np.zeros((len(actions[state_current]), len(states)))

        # rewards for shirk based on the action
        for action in range(len(actions[state_current])):

            reward_temp[action, 0:state_current+action+1] = (
                (len(states)-1-action) * reward_shirk)

        reward_func.append(reward_temp)

    return reward_func


def reward_final(states, reward_thr, reward_extra, thr, states_no):
    """
    when reward comes at final step -- again threshold at thr and extra rewards
    uptil states_NO (max number of states)
    in the course, thr=14 and max no of units = 22
    """
    total_reward_func_last = np.zeros(len(states))
    # np.zeros(len(states))
    # np.arange(0, states_no, 1)*reward_thr
    total_reward_func_last[thr:states_no] = (
        thr*reward_thr + np.arange(0, states_no-thr)*reward_extra)
    total_reward_func_last[states_no:] = (
        thr*reward_thr + (states_no-1-thr)*reward_extra)

    return total_reward_func_last


def effort(states, actions, effort_work):
    """
    immediate effort from actions
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, :] = action * effort_work

        effort_func.append(effort_temp)

    return effort_func


def effort_convex_concave(states, actions, effort_work, exponent):
    """
    immediate effort from actions, allowing not only linear but also concave
    and convex costs as functions of number of units done
    """

    effort_func = []
    for state_current in range(len(states)):

        effort_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            effort_temp[action, :] = (action**exponent) * effort_work

        effort_func.append(effort_temp)

    return effort_func


def T_uniform(states, actions):
    """
    transition function as a uniformly random process
    equal probability of next state for each action
    """

    T = []
    for state_current in range(len(states)):

        T_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            T_temp[action, state_current:state_current+action+1] = (
                np.full((action+1,), 1/(action+1))
            )

        T.append(T_temp)

    return T


def binomial_pmf(n, p, k):
    """
    returns binomial probability mass function given p = probability of
    success and n = number of trials, k = number of successes
    """

    if not isinstance(n, (int, np.int32, np.int64)):
        raise TypeError("Input must be an integer.")

    binomial_prob = comb(n, k) * p**k * (1-p)**(n-k)

    return binomial_prob


def T_binomial(states, actions, efficacy):
    """
    transition function as binomial number of successes with
    probability of success = efficacy and number of trials =
    number of units worked  (i.e., action)
    """

    T = []
    for state_current in range(len(states)):

        T_temp = np.zeros((len(actions[state_current]), len(states)))

        for i, action in enumerate(actions[state_current]):

            # T_temp[action, state_current:state_current+action+1] = (
            #     binom(action, efficacy).pmf(np.arange(action+1)))
            T_temp[action, state_current:state_current+action+1] = (
                binomial_pmf(action, efficacy, np.arange(action+1))
            )

        T.append(T_temp)

    return T


def T_binomial_decreasing(states, actions, horizon, efficacy):
    """
    time-decreasing binomial transition probabilities
    """

    T = []
    for i_timestep in range(horizon):
        T_t = []
        efficacy_t = efficacy * (1 - (i_timestep / horizon))

        for state_current in range(len(states)):

            T_temp = np.zeros((len(actions[state_current]), len(states)))

            for i, action in enumerate(actions[state_current]):

                T_temp[action, state_current:state_current+action+1] = (
                    binomial_pmf(action, efficacy_t, np.arange(action))
                )

            T_t.append(T_temp)
        T.append(T_t)
    return T


def deterministic_policy(a):
    p = np.where(a == np.max(a), 1, 0)
    return p / sum(p)


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p
