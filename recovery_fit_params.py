"""
implements model and parameter recovery for parameters and models fitted to
data. tells us if they are recoverable within the fitted range
"""

import numpy as np
import concurrent.futures
import constants
import recovery
import random

# %%
if __name__ == "__main__":

    np.random.seed(0)

    # generate data
    N_TRIALS = 15
    models = ['basic', 'efficacy_gap', 'convex_concave', 'immediate_basic',
              'diff_discounts', 'no_commitment']
    # load list of parameters and model from model fits
    fit_params = np.load('fit_params.npy', allow_pickle=True)
    # generate iterable list of input params and models
    input_lst = []

    # iterate through list, make a list of ALL parameters (incl. free params)
    # needed for generating data
    for i in range(len(fit_params)):

        # basic model
        if fit_params[i, 1] == 0:
            # parameters from fit
            (discount_factor,
             efficacy,
             effort_work) = fit_params[i, 0]

            # all input params + model identity, repeated 20 times
            for _ in range(20):
                input_lst.append([[constants.STATES, constants.ACTIONS,
                                   constants.HORIZON, constants.REWARD_THR,
                                   constants.REWARD_EXTRA,
                                   constants.REWARD_SHIRK, constants.BETA,
                                   discount_factor, efficacy, effort_work,
                                   N_TRIALS, constants.THR,
                                   constants.STATES_NO], 0])

        # efficacy gap
        elif fit_params[i, 1] == 1:

            (discount_factor,
             efficacy_assumed,
             efficacy_actual,
             effort_work) = fit_params[i, 0]

            for _ in range(20):
                input_lst.append([[constants.STATES, constants.ACTIONS,
                                   constants.HORIZON, constants.REWARD_THR,
                                   constants.REWARD_EXTRA,
                                   constants.REWARD_SHIRK, constants.BETA,
                                   discount_factor, efficacy_assumed,
                                   efficacy_actual, effort_work, N_TRIALS,
                                   constants.THR, constants.STATES_NO], 1])

        # convex concave
        elif fit_params[i, 1] == 2:

            (discount_factor,
             efficacy,
             effort_work,
             exponent) = fit_params[i, 0]

            for _ in range(20):
                input_lst.append([[constants.STATES, constants.ACTIONS,
                                   constants.HORIZON, constants.REWARD_THR,
                                   constants.REWARD_EXTRA,
                                   constants.REWARD_SHIRK, constants.BETA,
                                   discount_factor, efficacy, effort_work,
                                   exponent, N_TRIALS, constants.THR,
                                   constants.STATES_NO], 2])

        # immediate basic
        elif fit_params[i, 1] == 3:

            (discount_factor,
             efficacy,
             effort_work,
             exponent) = fit_params[i, 0]

            for _ in range(20):
                input_lst.append([[constants.STATES, constants.ACTIONS,
                                   constants.HORIZON, constants.REWARD_THR,
                                   constants.REWARD_EXTRA,
                                   constants.REWARD_SHIRK, constants.BETA,
                                   discount_factor, efficacy, effort_work,
                                   exponent, N_TRIALS, constants.THR,
                                   constants.STATES_NO], 3])

        # diff discounts
        elif fit_params[i, 1] == 4:

            (discount_factor_reward,
             discount_factor_cost,
             efficacy,
             effort_work) = fit_params[i, 0]

            for _ in range(20):
                input_lst.append([[constants.STATES, constants.ACTIONS,
                                   constants.HORIZON,
                                   constants.REWARD_THR_DIFF_DISCOUNTS,
                                   constants.REWARD_EXTRA_DIFF_DISCOUNTS,
                                   constants.REWARD_SHIRK,
                                   constants.BETA_DIFF_DISCOUNTS,
                                   discount_factor_reward,
                                   discount_factor_cost, efficacy, effort_work,
                                   N_TRIALS, constants.THR,
                                   constants.STATES_NO], 4])

        # no commitment
        elif fit_params[i, 1] == 5:

            (discount_factor,
             efficacy,
             effort_work,
             reward_interest) = fit_params[i, 0]

            for _ in range(20):
                input_lst.append([[constants.STATES_NO_COMMIT,
                                   constants.ACTIONS_BASE, constants.HORIZON,
                                   constants.REWARD_THR,
                                   constants.REWARD_EXTRA,
                                   constants.REWARD_SHIRK, constants.BETA,
                                   constants.P_STAY_LOW, constants.P_STAY_HIGH,
                                   discount_factor, efficacy, effort_work,
                                   reward_interest, N_TRIALS, constants.THR,
                                   constants.STATES_NO_NO_COMMIT], 5])
                
    # shuffle input list so that identical inputs are not in a row
    # when generating data parallely with seed, identical inputs generate same
    # trajectories
    random.Random(0).shuffle(input_lst)

    # recover model and params, parallelised
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(recovery.model_recovery, input_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    np.save('result_recovery_fit_params.npy', result)

    input_lst = [*input_lst]
    inputs = np.array(input_lst, dtype=object)
    np.save('input_recovery_fit_params.npy', inputs)
