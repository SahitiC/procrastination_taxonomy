"""
module to simulate data (trajectories) for each model given
input parameters and task structure
"""

import mdp_algms
import task_structure
import numpy as np


def softmax_policy(a, beta):
    c = a - np.max(a)
    p = np.exp(beta*c) / np.sum(np.exp(beta*c))
    return p


def gen_data_basic(states, actions, horizon, reward_unit, reward_shirk, beta,
                   discount_factor, efficacy, effort_work, n_trials,
                   states_no):
    """
    function to generate a trajectory of state and action sequences given
    parameters and reward, transition models of the basic model
    """

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    # reward delivered at the end of the semester
    total_reward_func_last = task_structure.reward_final_no_thr(
        states, reward_unit, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get tranistions
    T = task_structure.T_binomial(states, actions, efficacy)

    # get policy
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    # generate data - forward runs
    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions, initial_state, horizon, states,
            T, beta)
        data.append(s)

    return data


# function to generate a trajctory given parameters using efficacy-gap model
def gen_data_efficacy_gap(states, actions, horizon, reward_unit, reward_shirk,
                          beta, discount_factor, efficacy_assumed,
                          efficacy_actual, effort_work, n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the efficacy gap model; same as
    gen_data_basic except planning is based on assumed efficacy
    """

    # get reward function
    reward_func = task_structure.reward_no_immediate(
        states, actions, reward_shirk)

    effort_func = task_structure.effort(states, actions, effort_work)

    total_reward_func_last = task_structure.reward_final_no_thr(
        states, reward_unit, states_no)

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get transitions based on assumed efficacy
    T_assumed = task_structure.T_binomial(states, actions, efficacy_assumed)

    # get policy according to assumed transitions
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T_assumed)

    # get transition prob based on actual efficacy
    T_actual = task_structure.T_binomial(states, actions, efficacy_actual)

    # generate data - forward runs based on actual tranistion prob
    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions, initial_state, horizon, states,
            T_actual, beta)
        data.append(s)

    return data


# function to generate trajectory with non-linear costs (delayed rewards)
# and immediate rewards
def gen_data_convex_concave(states, actions, horizon, reward_unit,
                            reward_shirk, beta, discount_factor, efficacy,
                            effort_work, exponent, n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the convex concave model; same
    as gen_data_basic expect effort scales non-linearly with number of units
    """

    reward_func = task_structure.reward_immediate(
        states, actions, reward_shirk, reward_unit)

    effort_func = task_structure.effort_convex_concave(states, actions,
                                                       effort_work, exponent)

    total_reward_func_last = np.zeros(len(states))

    total_reward_func = []
    for state_current in range(len(states)):

        total_reward_func.append(reward_func[state_current]
                                 + effort_func[state_current])

    # get transition function
    T = task_structure.T_binomial(states, actions, efficacy)

    # get optimal policy according to task structure
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states, actions, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions, initial_state, horizon, states,
            T, beta)
        data.append(s)

    return data


def gen_data_diff_discounts(states, actions, horizon, reward_unit,
                            reward_shirk, beta, discount_factor_reward,
                            discount_factor_cost, efficacy, effort_work,
                            n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given
    parameters and reward, transition models of the diff-disc model
    """

    reward_func = task_structure.reward_immediate(
        states, actions, reward_shirk, reward_unit)

    effort_func = task_structure.effort(states, actions, effort_work)

    reward_func_last = np.zeros(len(states))
    effort_func_last = np.zeros(len(states))

    T = task_structure.T_binomial(states, actions, efficacy)

    V_opt_full, policy_opt_full, Q_values_full = (
        mdp_algms.find_optimal_policy_diff_discount_factors(
            states, actions, horizon, discount_factor_reward,
            discount_factor_cost, reward_func, effort_func, reward_func_last,
            effort_func_last, T)
    )

    # effective Q_values for the agent
    effective_Q = []
    for i_s in range(len(states)):
        Q_s_temp = []
        for i in range(horizon):
            Q_s_temp.append(Q_values_full[horizon-1-i][i_s][:, i])
        effective_Q.append(np.array(Q_s_temp).T)

    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, effective_Q, actions, initial_state, horizon,
            states, T, beta)
        data.append(s)

    return data


def gen_data_no_commitment(states, actions_base, horizon, reward_unit,
                           reward_shirk, beta, p_stay_low,
                           p_stay_high, discount_factor, efficacy, effort_work,
                           reward_interest, n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the no commitment model
    """

    states_no = len(states)

    # reward for completion
    reward_func_base = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_unit)

    # immediate interest rewards
    reward_func_interest = task_structure.reward_immediate(
        states[:int(states_no/2)], actions_base, 0, reward_interest)

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
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states[:int(states_no/2)])):

        temp = np.block([(1 - p_stay_high) * T_partial[state_current],
                         p_stay_high * T_partial[state_current]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
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

    initial_state = 0
    data = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions_all, initial_state, horizon,
            states, T, beta)
        s_unit = np.where(s > states_no/2 - 1, s-states_no/2, s)
        data.append(s_unit.astype(int))

    return data


def gen_data_fatigue(states_fatigue, actions_base, horizon, reward_unit,
                     reward_shirk, beta, p_low, p_high,
                     discount_factor, efficacy, effort_low,
                     effort_high, n_trials, states_no):
    """
    function to generate a trajectory of state and action sequences given 
    parameters and reward, transition models of the no commitment model
    """

    states_no = len(states_fatigue)

    # reward for completion
    reward_func_base = task_structure.reward_immediate(
        states_fatigue[:int(states_no/2)], actions_base, reward_shirk,
        reward_unit)

    # effort costs
    effort_func_low = task_structure.effort(states_fatigue[:int(states_no/2)],
                                            actions_base, effort_low)
    effort_func_high = task_structure.effort(states_fatigue[:int(states_no/2)],
                                             actions_base, effort_high)

    # total reward for low fatigue state = reward_base + effort_low
    total_reward_func_low = []
    for state_current in range(len(states_fatigue[:int(states_no/2)])):

        temp = reward_func_base[state_current] + effort_func_low[state_current]
        # replicate rewards for high reward states
        total_reward_func_low.append(np.block([temp, temp]))

    # total reward for high fatigue state = reward_base + effort_high
    total_reward_func_high = []
    for state_current in range(len(states_fatigue[:int(states_no/2)])):

        temp = reward_func_base[state_current] + \
            effort_func_high[state_current]
        # replicate rewards for high reward states
        total_reward_func_high.append(np.block([temp, temp]))

    total_reward_func = []
    total_reward_func.extend(total_reward_func_low)
    total_reward_func.extend(total_reward_func_high)

    total_reward_func_last = np.zeros(len(states_fatigue))

    # transition matrix based on efficacy and stay-switch probabilities
    T_partial = task_structure.T_binomial(states_fatigue[:int(states_no/2)],
                                          actions_base, efficacy)

    T_low_fatigue = np.column_stack(
        (np.linspace(1, p_low, int(states_no/2)),
         np.linspace(0, 1 - p_low, int(states_no/2))))

    T_high_fatigue = np.column_stack(
        (np.linspace(1 - p_high, 0, int(states_no/2)),
         np.linspace(p_high, 1, int(states_no/2))))

    T_low = []
    for state_current in range(len(states_fatigue[:int(states_no/2)])):
        temp = np.block(
            [T_partial[state_current]
             * T_low_fatigue[:int(states_no/2)-state_current, 0:1],
             T_partial[state_current]
             * T_low_fatigue[:int(states_no/2)-state_current, 1:2]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_low.append(temp)

    T_high = []
    for state_current in range(len(states_fatigue[:int(states_no/2)])):
        temp = np.block(
            [T_partial[state_current]
             * T_high_fatigue[:int(states_no/2)-state_current, 0:1],
             T_partial[state_current]
             * T_high_fatigue[:int(states_no/2)-state_current, 1:2]])
        assert (np.round(np.sum(temp, axis=1), 6) == 1).all()
        T_high.append(temp)

    T = []
    T.extend(T_low)
    T.extend(T_high)

    # optimal policy based on task structure
    actions_all = actions_base.copy()
    # same actions available for low and high reward states: so repeat
    actions_all.extend(actions_base)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards(
        states_fatigue, actions_all, horizon, discount_factor,
        total_reward_func, total_reward_func_last, T)

    initial_state = 0
    data = []
    data_s = []
    data_actions = []
    for i_trials in range(n_trials):

        s, a = mdp_algms.forward_runs_prob(
            softmax_policy, Q_values, actions_all, initial_state, horizon,
            states_fatigue, T, beta)
        s_unit = np.where(s > states_no/2 - 1, s-states_no/2, s)
        data.append(s_unit.astype(int))
        data_s.append(s)
        data_actions.append(a)

    return data, data_s, data_actions
