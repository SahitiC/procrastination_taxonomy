"""
implements model and parameter recovery for all six models. parameters are
chosen from a range of feasible values (agnostic to the parameters fitted to
the data). data is simulated from randomly chosen parameters for each model.
all models are fitted to data, best model and parameters are chosen 
"""

import numpy as np
import likelihoods
import gen_data
import concurrent.futures
import constants

# %%


def model_recovery(inputs):

    models = ['basic', 'efficacy_gap', 'convex_concave', 'immediate_basic',
              'diff_discounts', 'no_commitment']
    # model and params to generate data
    model = models[inputs[1]]
    params = inputs[0]
    gen_data_model = getattr(gen_data, 'gen_data_'+model)
    data = gen_data_model(*params)

    # fit each model to data and recover params
    result_basic = likelihoods.maximum_likelihood_estimate_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data)

    result_efficacy_gap = likelihoods.maximum_likelihood_estimate_efficacy_gap(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data)

    result_conv_conc = likelihoods.maximum_likelihood_estimate_convex_concave(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data)

    result_imm_basic = likelihoods.maximum_likelihood_estimate_immediate_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, constants.THR, constants.STATES_NO, data)

    result_diff_disc = likelihoods.maximum_likelihood_estimate_diff_discounts(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR_DIFF_DISCOUNTS,
        constants.REWARD_EXTRA_DIFF_DISCOUNTS, constants.REWARD_SHIRK,
        constants.BETA_DIFF_DISCOUNTS, constants.THR, constants.STATES_NO,
        data)

    result_no_commit = likelihoods.maximum_likelihood_estimate_no_commitment(
        constants.STATES_NO_COMMIT, constants.INTEREST_STATES,
        constants.ACTIONS_BASE, constants.HORIZON, constants.REWARD_THR,
        constants.REWARD_EXTRA, constants.REWARD_SHIRK, constants.BETA,
        constants.P_STAY_LOW, constants.P_STAY_HIGH, constants.THR,
        constants.STATES_NO_NO_COMMIT, data)

    results = [result_basic, result_efficacy_gap, result_conv_conc,
               result_imm_basic, result_diff_disc, result_no_commit]

    # find best model, what are the parameters?
    nllkhds = [result_basic.fun, result_efficacy_gap.fun,
               result_conv_conc.fun, result_imm_basic.fun,
               result_diff_disc.fun, result_no_commit.fun]
    params = [result_basic.x, result_efficacy_gap.x,
              result_conv_conc.x, result_imm_basic.x,
              result_diff_disc.x, result_no_commit.x]

    return [nllkhds, params]


# %%
if __name__ == "__main__":

    np.random.seed(0)

    # generate data
    N = 500  # no. of param sets per model type to recover
    N_TRIALS = 15  # no. of trials
    # no. of free params for each of these models
    free_param_no = [3, 4, 4, 4, 4, 4]

    # generate iterable list of input params and models
    input_lst = []

    # basic model
    for i in range(N):
        # generate random parameters
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.5)

        # all input params and model identity
        input_lst.append([[constants.STATES, constants.ACTIONS,
                           constants.HORIZON, constants.REWARD_THR,
                           constants.REWARD_EXTRA, constants.REWARD_SHIRK,
                           constants.BETA, discount_factor, efficacy,
                           effort_work, N_TRIALS, constants.THR,
                           constants.STATES_NO], 0])

    # efficacy gap
    for i in range(N):
        # generate random parameters
        discount_factor = np.random.uniform(0.2, 1)
        efficacy_assumed = np.random.uniform(0.25, 0.9)
        # sample actual efficacies such that they are diff from effic assumed
        prob_coin_flip = ((efficacy_assumed-0.25)
                          / (efficacy_assumed-0.25 + 1-efficacy_assumed-0.1))
        coin_flip = np.random.binomial(1, prob_coin_flip)
        if coin_flip == 1:
            efficacy_actual = np.random.uniform(0.25, efficacy_assumed-0.1)
        elif coin_flip == 0:
            efficacy_actual = np.random.uniform(efficacy_assumed+0.1, 1)

        effort_work = -1 * np.random.exponential(0.5)

        input_lst.append([[constants.STATES, constants.ACTIONS,
                           constants.HORIZON, constants.REWARD_THR,
                           constants.REWARD_EXTRA, constants.REWARD_SHIRK,
                           constants.BETA, discount_factor, efficacy_assumed,
                           efficacy_actual, effort_work, N_TRIALS,
                           constants.THR, constants.STATES_NO], 1])

    # convex concave
    for i in range(N):
        # generate random parameters
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.25, 1)
        effort_work = -1 * np.random.exponential(0.5)
        exponent = np.random.gamma(2.5, 0.5)

        input_lst.append([[constants.STATES, constants.ACTIONS,
                           constants.HORIZON, constants.REWARD_THR,
                           constants.REWARD_EXTRA, constants.REWARD_SHIRK,
                           constants.BETA, discount_factor, efficacy,
                           effort_work, exponent, N_TRIALS, constants.THR,
                           constants.STATES_NO], 2])

    # immediate basic
    for i in range(N):
        # generate random parameters
        discount_factor = np.random.uniform(0.2, 1)
        efficacy = np.random.uniform(0.2, 1)
        effort_work = -1 * np.random.exponential(0.5)
        exponent = np.random.gamma(2.5, 0.5)

        input_lst.append([[constants.STATES, constants.ACTIONS,
                           constants.HORIZON, constants.REWARD_THR,
                           constants.REWARD_EXTRA, constants.REWARD_SHIRK,
                           constants.BETA, discount_factor, efficacy,
                           effort_work, exponent, N_TRIALS, constants.THR,
                           constants.STATES_NO], 3])

    # diff discounts
    for i in range(N):
        # generate random parameters
        discount_factor_reward = np.random.uniform(0.2, 1)
        # sample only those < disc_reward-0.1
        discount_factor_cost = np.random.uniform(0.2,
                                                 discount_factor_reward-0.1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.5)

        input_lst.append([[constants.STATES, constants.ACTIONS,
                           constants.HORIZON,
                           constants.REWARD_THR_DIFF_DISCOUNTS,
                           constants.REWARD_EXTRA_DIFF_DISCOUNTS,
                           constants.REWARD_SHIRK,
                           constants.BETA_DIFF_DISCOUNTS,
                           discount_factor_reward,
                           discount_factor_cost, efficacy, effort_work,
                           N_TRIALS, constants.THR, constants.STATES_NO], 4])

    # no commitment
    for i in range(N):
        # generate random parameters
        discount_factor = np.random.uniform(0.6, 1)
        efficacy = np.random.uniform(0.35, 1)
        effort_work = -1 * np.random.exponential(0.67)
        reward_interest = np.random.gamma(3, 1)

        input_lst.append([[constants.STATES_NO_COMMIT,
                           constants.ACTIONS_BASE, constants.HORIZON,
                           constants.REWARD_THR, constants.REWARD_EXTRA,
                           constants.REWARD_SHIRK, constants.BETA,
                           constants.P_STAY_LOW, constants.P_STAY_HIGH,
                           discount_factor, efficacy, effort_work,
                           reward_interest, N_TRIALS, constants.THR,
                           constants.STATES_NO_NO_COMMIT], 5])

    # recover model and params, parallelised
    with concurrent.futures.ProcessPoolExecutor() as executor:
        result_lst = executor.map(model_recovery, input_lst)

    result_lst = [*result_lst]
    result = np.array(result_lst, dtype=object)
    np.save('result_recovery.npy', result)

    input_lst = [*input_lst]
    inputs = np.array(input_lst, dtype=object)
    np.save('inputs_recovery.npy', inputs)
