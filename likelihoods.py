"""
module to calculate likelihood of data under each of the models and maximise
log likelihood (minimise negative log likelihood) to find best fitting params
"""

import task_structure
import mdp_algms
import numpy as np
from scipy.optimize import minimize


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def calculate_likelihood(data, Q_values, beta, T, actions):
    """
    calculate likelihood of data under model given optimal Q_values, beta,
    transitions and actions available
    """
    nllkhd = 0

    for i_trial in range(len(data)):

        for i_time in range(len(data[i_trial])-1):

            partial = 0
            # enumerate over all posible actions for the observed state
            for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                partial += (
                    softmax_policy(Q_values[data[i_trial][i_time]]
                                   [:, i_time], beta)[action]
                    * T[data[i_trial][i_time]][action][
                        data[i_trial][i_time+1]])

            nllkhd = nllkhd - np.log(partial)

    return nllkhd


def calculate_likelihood_interest_rewards(data, Q_values, beta, T, p_stay,
                                          actions, interest_states):
    """
    calculate likelihood of data under interest reward model given 
    optimal Q_values, beta, transitions, probability of staying in low and
    high states, and actions available
    """
    nllkhd = 0

    for i_trial in range(len(data)):

        # marginal prob of interest rewards at very first time step
        p_interest = np.zeros(len(interest_states))
        for i_a, action in enumerate(actions[data[i_trial][0]]):

            p_interest += (p_stay[0, :]  # assume 1st interest state = 0 (low)
                           * softmax_policy(Q_values[0][data[i_trial][0]]
                                            [:, 0], beta)[action]
                           * T[data[i_trial][0]][action][data[i_trial][1]])

        # marginal prob for rest of time steps
        for i_time in range(1, len(data[i_trial])-1):

            partial = np.zeros(len(interest_states))

            # enumerate over all possible interest states
            for i_i, interest_state in enumerate(interest_states):

                # enumerate over all possible actions for the observed state
                for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                    partial += (
                        p_stay[interest_state, :]
                        * softmax_policy(Q_values[interest_state]
                                         [data[i_trial][i_time]]
                                         [:, i_time], beta)[action]
                        * T[data[i_trial][i_time]][action][
                            data[i_trial][i_time+1]]
                        * p_interest[interest_state])

            # the above calculation results in a marginal prob over the (two)
            # possible interest states, which'll be added up in next iteration
            p_interest = partial

        # final prob is over the two interest states, so must be added up
        nllkhd = nllkhd - np.log(np.sum(p_interest))

    return nllkhd


def likelihood_basic_model(x,
                           states, actions, horizon,
                           reward_thr, reward_extra, reward_shirk,
                           beta, thr, states_no, data):
    """
    implement likelihood calculation for basic model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_efficacy_gap_model(x,
                                  states, actions, horizon,
                                  reward_thr, reward_extra, reward_shirk,
                                  beta, thr, states_no, data):
    """
    implement likelihood calculation for efficacy gap model
    """

    discount_factor = x[0]
    efficacy_assumed = x[1]
    efficacy_actual = x[2]
    effort_work = x[3]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T_assumed = task_structure.T_binomial(states, actions, efficacy_assumed)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T_assumed)

    T_actual = task_structure.T_binomial(states, actions, efficacy_actual)

    nllkhd = calculate_likelihood(data, Q_values, beta, T_actual, actions)

    return nllkhd


def likelihood_convex_concave_model(x,
                                    states, actions, horizon,
                                    reward_thr, reward_extra, reward_shirk,
                                    beta, thr, states_no, data):
    """
    implement likelihood calculation for convex concave model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]
    exponent = x[3]

    # define task structure
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort_convex_concave(states, actions,
                                                       effort_work, exponent)

    total_reward_func_last = task_structure.reward_final(
        states, reward_thr, reward_extra, thr, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_immediate_basic_model(x,
                                     states, actions, horizon,
                                     reward_thr, reward_extra, reward_shirk,
                                     beta, thr, states_no, data):
    """
    implement likelihood calculation for immediate basic model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]
    exponent = x[3]

    # define task structure
    reward_func = task_structure.reward_threshold(
        states, actions, reward_shirk, reward_thr, reward_extra, thr,
        states_no)

    effort_func = task_structure.effort_convex_concave(states, actions,
                                                       effort_work, exponent)

    total_reward_func_last = np.zeros(len(states))

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    T = task_structure.T_binomial(states, actions, efficacy)

    # optimal policy under assumed efficacy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    nllkhd = calculate_likelihood(data, Q_values, beta, T, actions)

    return nllkhd


def likelihood_diff_discounts_model(
        x, states, actions, horizon, reward_thr, reward_extra,
        reward_shirk, beta, thr, states_no, data):
    """
    implement likelihood calculation for diff discount model
    """

    discount_factor_reward = x[0]
    discount_factor_cost = x[1]
    efficacy = x[2]
    effort_work = x[3]
    # reward_shirk = x[2]

    reward_func = task_structure.reward_threshold(
        states, actions, reward_shirk, reward_thr, reward_extra, thr,
        states_no)

    effort_func = task_structure.effort(states, actions, effort_work)

    reward_func_last = np.zeros(len(states))
    effort_func_last = np.zeros(len(states))

    T = task_structure.T_binomial(states, actions, efficacy)

    V_opt_full, policy_opt_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            states, actions, horizon, discount_factor_reward,
            discount_factor_cost, reward_func, effort_func, reward_func_last,
            effort_func_last, T))

    # effective Q_values for the agent
    effective_Q = []
    for i_s in range(len(states)):
        Q_s_temp = []
        for i in range(horizon):
            Q_s_temp.append(Q_values_full[horizon-1-i][i_s][:, i])
        effective_Q.append(np.array(Q_s_temp).T)

    nllkhd = calculate_likelihood(data, effective_Q, beta, T, actions)

    return nllkhd


def likelihood_no_commitment_model(
        x, states, interest_states, actions_base, horizon, p_stay_low,
        p_stay_high, reward_thr, reward_extra, reward_shirk,
        beta, thr, states_no, data):
    """
    implement likelihood calculation for no commit model
    """

    discount_factor = x[0]
    efficacy = x[1]
    effort_work = x[2]
    reward_interest = x[3]

    states_no = len(states)

    # reward for completion
    reward_func_base = task_structure.reward_threshold(
        states[:int(states_no/2)], actions_base, reward_shirk,
        reward_thr, reward_extra, thr, states_no)

    # immediate interest rewards
    reward_func_interest = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_interest,
        reward_interest)

    # effort costs
    effort_func = task_structure.effort(states[:int(states_no/2)],
                                        actions_base, effort_work)

    # total reward for low reward state = reward_base + effort
    total_reward_func_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = reward_func_base[state_current] + effort_func[state_current]
        # replicate rewards for high reward states
        total_reward_func_low.append(np.block([temp, temp]))

    # total reward for high reward state = reward_base+interest rewards+effort
    total_reward_func_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = (reward_func_base[state_current]
                + reward_func_interest[state_current]
                + effort_func[state_current])
        total_reward_func_high.append(np.block([temp, temp]))

    total_reward_func = []
    total_reward_func.extend(total_reward_func_low)
    total_reward_func.extend(total_reward_func_high)

    total_reward_func_last = np.zeros(len(states))

    # tranistion matrix based on efficacy and stay-switch probabilities
    T_partial = task_structure.T_binomial(states[:int(states_no/2)],
                                          actions_base, efficacy)
    T_low = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([p_stay_low * T_partial[state_current],
                         (1 - p_stay_low) * T_partial[state_current]])
        # assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([(1 - p_stay_high) * T_partial[state_current],
                         p_stay_high * T_partial[state_current]])
        # assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_high.append(temp)

    T = []
    T.extend(T_low)
    T.extend(T_high)

    # optimal policy based on task structure
    actions_all = actions_base.copy()
    # same actions available for low and high reward states: so repeat
    actions_all.extend(actions_base)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions_all, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # adapt some quantities for llkhd calculating function
    p_stay = np.array([[p_stay_low, 1-p_stay_low],
                       [1-p_stay_high, p_stay_high]])

    Q_values_unstacked = np.array([Q_values[:int(states_no/2)],
                                   Q_values[int(states_no/2):]])

    nllkhd = calculate_likelihood_interest_rewards(
        data, Q_values_unstacked, beta, T_partial, p_stay, actions_base,
        interest_states)

    return nllkhd


def maximum_likelihood_estimate_basic(states, actions, horizon, reward_thr,
                                      reward_extra, reward_shirk, beta,
                                      thr, states_no, data,
                                      true_params=None, initial_real=0,
                                      verbose=0):
    """
    maximise likelihood of data under basic model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_basic_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, thr, states_no, data),
                                bounds=((0, 1), (0, 1), (None, 0)))
        nllkhd = likelihood_basic_model(
            final_result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, efficacy,"
                  f" effort_work = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)
        # exponential distribution for beta with lambda = 1 or scale = 1
        # following Wilson and Collins 2019:
        # beta = np.random.exponential(2)
        # reward_shirk = np.random.exponential(0.5)
        effort_work = -1 * np.random.exponential(0.5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_basic_model,
                          x0=[discount_factor, efficacy, effort_work],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, thr, states_no, data),
                          bounds=((0, 1), (0, 1), (None, 0)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_basic_model(
            result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, efficacy,"
                    f" effort_work = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_efficacy_gap(
        states, actions, horizon, reward_thr, reward_extra, reward_shirk, beta,
        thr, states_no, data, true_params=None, initial_real=0, verbose=0):
    """
    maximise likelihood of data under efficacy gap model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_efficacy_gap_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, thr, states_no, data),
                                bounds=((0, 1), (0, 1), (0, 1), (None, 0)))
        nllkhd = likelihood_efficacy_gap_model(
            final_result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, efficacy_assumed,"
                  f" efficacy_gap, effort_work = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy_assumed = np.random.uniform(0, 1)
        efficacy_actual = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_efficacy_gap_model,
                          x0=[discount_factor, efficacy_assumed,
                              efficacy_actual, effort_work],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, thr, states_no, data),
                          bounds=((0, 1), (0, 1), (0, 1), (None, 0)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_efficacy_gap_model(
            result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, efficacy_assumed,"
                    f" efficacy_actual, effort_work = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_convex_concave(
        states, actions, horizon, reward_thr, reward_extra, reward_shirk,
        beta, thr, states_no, data, true_params=None, initial_real=0,
        verbose=0):
    """
    maximise likelihood of data under conv concave model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_convex_concave_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, thr, states_no, data),
                                bounds=((0, 1), (0, 1), (None, 0), (0, None)))
        nllkhd = likelihood_convex_concave_model(
            final_result.x, states, actions, horizon, reward_thr,
            reward_extra, reward_shirk, beta, thr, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, effort_work,"
                  f" exponent = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        exponent = np.random.gamma(2.5, 0.5)
        efficacy = np.random.uniform(0, 1)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_convex_concave_model,
                          x0=[discount_factor, efficacy, effort_work, exponent],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, thr, states_no, data),
                          bounds=((0, 1), (0, 1), (None, 0), (0, None)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_convex_concave_model(
            result.x, states, actions, horizon, reward_thr,
            reward_extra, reward_shirk, beta, thr, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, effort_work,"
                    f" exponent = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_immediate_basic(
        states, actions, horizon, reward_thr, reward_extra, reward_shirk,
        beta, thr, states_no, data, true_params=None, initial_real=0,
        verbose=0):
    """
    maximise likelihood of data under  immediate basic model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_immediate_basic_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, thr, states_no, data),
                                bounds=((0, 1), (0, 1), (None, 0), (0, None)))
        nllkhd = likelihood_immediate_basic_model(
            final_result.x, states, actions, horizon, reward_thr,
            reward_extra, reward_shirk, beta, thr, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor, effort_work,"
                  f" exponent, efficacy = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        exponent = np.random.gamma(2.5, 0.5)
        efficacy = np.random.uniform(0, 1)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_immediate_basic_model,
                          x0=[discount_factor, efficacy, effort_work, exponent],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, thr, states_no, data),
                          bounds=((0, 1), (0, 1), (None, 0), (0, None)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_immediate_basic_model(
            result.x, states, actions, horizon, reward_thr,
            reward_extra, reward_shirk, beta, thr, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print(
                    "current estimate for discount_factor, effort_work,"
                    f" exponent, efficacy = {final_result.x} "
                    f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_diff_discounts(
        states, actions, horizon, reward_thr, reward_extra,
        reward_shirk, beta, thr, states_no, data, true_params=None,
        initial_real=0, verbose=0):
    """
    maximise likelihood of data under diff-disc model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_diff_discounts_model,
                                x0=true_params,
                                args=(states, actions, horizon,
                                      reward_thr, reward_extra, reward_shirk,
                                      beta, thr, states_no, data),
                                bounds=((0, 1), (0, 1), (0, 1), (None, 0)))
        nllkhd = likelihood_diff_discounts_model(
            final_result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor_reward, "
                  f"discount_factor_cost, reward_shirk, effort_work, "
                  f"efficacy = {final_result.x}"
                  f"with neg log likelihood = {nllkhd}")

    # repeat likelihood-based optimisation for different initial values
    for i in range(20):

        # set initial value for params (random draws)
        discount_factor_reward = np.random.uniform(0, 1)
        discount_factor_cost = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        # reward_shirk = np.random.exponential(0.5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_diff_discounts_model,
                          x0=[discount_factor_reward, discount_factor_cost,
                              efficacy, effort_work],
                          args=(states, actions, horizon,
                                reward_thr, reward_extra, reward_shirk,
                                beta, thr, states_no, data),
                          bounds=((0, 1), (0, 1), (0, 1), (None, 0)))

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_diff_discounts_model(
            result.x, states, actions, horizon, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print("with initial point = true param "
                      "current estimate for discount_factor_reward, "
                      f"discount_factor_cost, reward_shirk, effort_work, "
                      f"efficacy = {final_result.x}"
                      f"with neg log likelihood = {nllkhd}")

    return final_result


def maximum_likelihood_estimate_no_commitment(
        states, interest_states, actions_base, horizon, reward_thr,
        reward_extra, reward_shirk, beta, p_stay_low, p_stay_high, thr,
        states_no, data, true_params=None, initial_real=0, verbose=0):
    """
    maximise likelihood of data under bno commit model parameters using 
    scipy.optimize
    initial_real: whether to include true parameter as an initial point in 
    optimisation procedure
    verbose: whether to print current estimate and likelihood
    """

    nllkhd = np.inf

    # find optimal estimate starting with initial point = true params
    if initial_real == 1:
        final_result = minimize(likelihood_no_commitment_model,
                                x0=true_params,
                                args=(states, interest_states, actions_base,
                                      horizon, p_stay_low, p_stay_high,
                                      reward_thr, reward_extra,
                                      reward_shirk, beta, thr, states_no,
                                      data),
                                bounds=((0, 1), (0, 1), (None, 0), (0, None)))
        # method='Powell')
        nllkhd = likelihood_no_commitment_model(
            final_result.x, states, interest_states, actions_base, horizon,
            p_stay_low, p_stay_high, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)
        if verbose == 1:
            print("with initial point = true param "
                  "current estimate for discount_factor,"
                  f" efficacy = {final_result.x} "
                  f"with neg log likelihood = {nllkhd}")

    for i in range(20):

        # set initial value for params (random draws)
        discount_factor = np.random.uniform(0, 1)
        efficacy = np.random.uniform(0, 1)
        effort_work = -1 * np.random.exponential(0.5)
        reward_interest = np.random.exponential(5)

        # minimise nllkhd with initial value to get param estimate
        result = minimize(likelihood_no_commitment_model,
                          x0=[discount_factor, efficacy, effort_work,
                              reward_interest],
                          args=(states, interest_states, actions_base,
                                horizon, p_stay_low, p_stay_high,
                                reward_thr, reward_extra,  # reward_interest,
                                reward_shirk, beta, thr, states_no, data),
                          bounds=((0, 1), (0, 1), (None, 0), (0, None)))
        # method='Powell')

        # whats the neg log likelhood of data under param estimate
        nllkhd_result = likelihood_no_commitment_model(
            result.x, states, interest_states, actions_base, horizon,
            p_stay_low, p_stay_high, reward_thr, reward_extra,
            reward_shirk, beta, thr, states_no, data)

        # is it better than previous estimate
        if nllkhd_result < nllkhd:

            nllkhd = nllkhd_result
            final_result = result
            if verbose == 1:
                print("with initial point = true param "
                      "current estimate for discount_factor "
                      f"efficacy = {final_result.x}"
                      f"with neg log likelihood = {nllkhd}")

    return final_result
