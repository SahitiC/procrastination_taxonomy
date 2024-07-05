import gen_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants
import task_structure
import seaborn as sns
import random
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3

# %%


def calculate_likelihood_null(data, states, actions):
    """
    likelihood of data under random model
    for the purpose of calculating pseudo R2
    """

    nllkhd = 0
    # T_uniform = task_structure.T_uniform(states, actions)

    for i_trial in range(len(data)):

        for i_time in range(len(data[i_trial])-1):

            partial = 0
            # transition probability with random efficacy
            efficacy = np.random.uniform(0, 1, 1)
            T = task_structure.T_binomial(states, actions, efficacy)

            # enumerate over all posible actions for the observed state
            for i_a, action in enumerate(actions[data[i_trial][i_time]]):

                # uniform prob for each action
                partial += (
                    1/len(actions[data[i_trial][i_time]])
                    * T[data[i_trial][i_time]][action][
                        data[i_trial][i_time+1]])

            nllkhd = nllkhd - np.log(partial)

    return nllkhd


def plot_trajectories(trajectories, color, lwidth_mean, lwidth_sample):
    """
    """
    mean = np.mean(trajectories, axis=0)

    plt.plot(mean, color=color, linewidth=lwidth_mean)
    for i in range(len(trajectories)):
        plt.plot(trajectories[i], color=color,
                 linewidth=lwidth_sample, linestyle='dashed')
    sns.despine()


# %%
# import results
free_param_no = [3, 4, 4, 4, 4, 4]  # no. of free params for each model
result_fit = np.load('result.npy', allow_pickle=True)
data_to_fit_lst = np.load('data_to_fit_lst.npy', allow_pickle=True)

# %%
for i in range(len(data_to_fit_lst)):
    plt.figure(figsize=(5, 4))
    for j in range(len(data_to_fit_lst[i])):
        plt.plot(data_to_fit_lst[i][j])

# %%
# get fitted parameters, calculate AIC, BIC, pseudo-R2
for cluster in range(len(data_to_fit_lst)):

    nllkhd_null = calculate_likelihood_null(
        data_to_fit_lst[cluster], constants.STATES, constants.ACTIONS)

    # pseudo-R2
    print(f'pseudo-r2={1 - (result_fit[cluster, 0, :] / nllkhd_null)}')

    # BIC
    print(
        f'BIC={2*result_fit[cluster, 0, :] + np.array(free_param_no) * np.log(len(data_to_fit_lst[cluster]))}')

    # AIC
    print(
        f'AIC={2*result_fit[cluster, 0, :] + np.array(free_param_no) * 2}')

# %%
# save the fitted params
fit_params = result_fit[:, 1, :].flatten()
# 6 models, 8 clusters
fit_params = np.vstack((fit_params,
                        np.tile(np.arange(0, 6, 1), 8)))
fit_params = fit_params.T
np.save('fit_params.npy', fit_params)

# %%
# code for average trajectory plots
# plt.errorbar(np.arange(constants.HORIZON),
#              np.mean(data, axis=0)[1:]/2,
#              yerr=np.std(data, axis=0)[1:]/2,
#              marker='o',
#              linestyle='--',
#              label='model')

# plt.errorbar(np.arange(constants.HORIZON),
#              np.mean(data_to_fit_lst[cluster], axis=0)[1:]/2,
#              yerr=np.std(data_to_fit_lst[cluster], axis=0)[1:]/2,
#              marker='o',
#              linestyle='--',
#              label='data')
# plt.xlabel('time (weeks)')
# plt.ylabel('rearch houurs \n completed')
# plt.legend(loc='upper left', fontsize=18)
# sns.despine()


# %%
# compare cluster trajectories with trajectories simulated from fitted models
random.seed(0)

cluster = 7
# no. of trials = no. of trajectories in the cluster
n = len(data_to_fit_lst[cluster])

# plot cluster
plt.figure(figsize=(4, 4), dpi=300)
plot_trajectories(data_to_fit_lst[cluster], 'gray', 2, 0.5)

# basic
plt.figure(figsize=(4, 4), dpi=300)
params = result_fit[cluster, 1, 0]
data = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, discount_factor=params[0], efficacy=params[1],
    effort_work=params[2], n_trials=n, thr=constants.THR,
    states_no=constants.STATES_NO)
plot_trajectories(data, 'gray', 2, 0.5)

# efficacy gap
plt.figure(figsize=(4, 4), dpi=300)
params = result_fit[cluster, 1, 1]
data = gen_data.gen_data_efficacy_gap(
    constants.STATES, constants.ACTIONS,  constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, discount_factor=params[0], efficacy_assumed=params[1],
    efficacy_actual=params[2], effort_work=params[3], n_trials=n,
    thr=constants.THR, states_no=constants.STATES_NO)
plot_trajectories(data, 'gray', 2, 0.5)

# convex concav
plt.figure(figsize=(4, 4), dpi=300)
params = result_fit[cluster, 1, 2]
data = gen_data.gen_data_convex_concave(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, discount_factor=params[0], efficacy=params[3],
    effort_work=params[1], exponent=params[2], n_trials=n, thr=constants.THR,
    states_no=constants.STATES_NO)
plot_trajectories(data, 'gray', 2, 0.5)

# imm basic
plt.figure(figsize=(4, 4), dpi=300)
params = result_fit[cluster, 1, 3]
data = gen_data.gen_data_immediate_basic(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, discount_factor=params[0], efficacy=params[3],
    effort_work=params[1], exponent=params[2], n_trials=n, thr=constants.THR,
    states_no=constants.STATES_NO)
plot_trajectories(data, 'gray', 2, 0.5)

# different discount
plt.figure(figsize=(4, 4), dpi=300)
params = result_fit[cluster, 1, 4]
data = gen_data.gen_data_diff_discounts(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR_DIFF_DISCOUNTS, constants.REWARD_EXTRA_DIFF_DISCOUNTS,
    constants.REWARD_SHIRK, constants.BETA_DIFF_DISCOUNTS,
    discount_factor_reward=params[0], discount_factor_cost=params[1],
    efficacy=params[2], effort_work=params[3], n_trials=n, thr=constants.THR,
    states_no=constants.STATES_NO)
plot_trajectories(data, 'gray', 2, 0.5)

# no commit
plt.figure(figsize=(4, 4), dpi=300)
params = result_fit[cluster, 1, 5]
data = gen_data.gen_data_no_commitment(
    constants.STATES_NO_COMMIT, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, constants.P_STAY_LOW, constants.P_STAY_HIGH,
    discount_factor=params[0], efficacy=params[1], effort_work=params[2],
    reward_interest=params[3], n_trials=n, thr=constants.THR,
    states_no=constants.STATES_NO)
plot_trajectories(data, 'gray', 2, 0.5)
