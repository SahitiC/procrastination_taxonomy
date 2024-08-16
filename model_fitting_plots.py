import gen_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import constants
import task_structure
import likelihoods
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
    plt.xticks([0, 7, 15])
    plt.yticks([0, 11, 22])
    plt.ylim(-1, 22)
    sns.despine()


def distance(vector1, vector2, variances):

    distance = 0
    for dimension in range(len(vector1)):

        distance = (distance
                    + ((vector1[dimension] - vector2[dimension])**2)
                    / variances[dimension])

    return distance


def get_distance_matrix(vectors):
    """
    finds pairwise distance between vectors in a list of vectors; vectors are
    weighted by the total variance in each of the vector dimensions

    params:
        vectors (list): list of vectors for which distance matrix is computed

    returns:
        distance_matrix(ndarray): matrix of pairwise distances between vectors;
        is symmetric and diagonal = 0
    """

    vector_no = len(vectors)  # no. of vectors
    variances = np.var(vectors, axis=0)
    distance_matrix = np.full((vector_no, vector_no), -1.1)
    for i in range(vector_no):
        for j in range(vector_no):

            distance_matrix[i, j] = distance(vectors[i, :], vectors[j, :],
                                             variances)
    return distance_matrix


# %% import results
free_param_no = [3, 4, 4, 4, 4, 4]  # no. of free params for each model
result_fit = np.load('result.npy', allow_pickle=True)
data_to_fit_lst = np.load('data_to_fit_lst.npy', allow_pickle=True)

np.random.seed(0)

# %%
# since conv-conc did not fit well to cluster 5 (it should do atleast as
# good as basic), re-run fitting for this

result_conv_conc = likelihoods.maximum_likelihood_estimate_convex_concave(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, constants.THR, constants.STATES_NO, data_to_fit_lst[4])

result_fit[4, 0, 2] = result_conv_conc.fun
result_fit[4, 1, 2] = result_conv_conc.x

# similarly for cluster 7, basic model, eff_gap model fit is much better
# and both efficacy_assumed and real are the same (= basic model)
# so basic model fit should be atleast as good

result_basic = likelihoods.maximum_likelihood_estimate_basic(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
    constants.BETA, constants.THR, constants.STATES_NO, data_to_fit_lst[7])

result_fit[7, 0, 0] = result_basic.fun
result_fit[7, 1, 0] = result_basic.x

# %% rearrange order of clusters to the order in the paper
new_order = [1, 7, 4, 6, 0, 5, 2, 3]
data_to_fit_lst = data_to_fit_lst[new_order]
result_fit = result_fit[new_order]

# %% get fitted parameters, calculate AIC, BIC, pseudo-R2

metrics = np.zeros((8, 6, 3))
for cluster in range(len(data_to_fit_lst)):

    nllkhd_null = calculate_likelihood_null(
        data_to_fit_lst[cluster], constants.STATES, constants.ACTIONS)

    # pseudo-R2
    metrics[cluster, :, 0] = 1 - (result_fit[cluster, 0, :] / nllkhd_null)

    # BIC
    metrics[cluster, :, 1] = (2*result_fit[cluster, 0, :]
                              + np.array(free_param_no)
                              * np.log(len(data_to_fit_lst[cluster])))
    # AIC
    metrics[cluster, :, 2] = (2*result_fit[cluster, 0, :]
                              + np.array(free_param_no) * 2)

# %% save the fitted params

fit_params = result_fit[:, 1, :].flatten()
# 6 models, 8 clusters
fit_params = np.vstack((fit_params,
                        np.tile(np.arange(0, 6, 1), 8)))
fit_params = fit_params.T
np.save('fit_params.npy', fit_params)

# %% compare cluster trajectories with trajectories simulated from fitted models

for cluster in range(8):

    # no. of trials = no. of trajectories in the cluster
    n = len(data_to_fit_lst[cluster])

    # plot cluster
    plt.figure(figsize=(4, 4), dpi=300)
    plot_trajectories(data_to_fit_lst[cluster], 'gray', 3, 1)
    # plt.xlabel('time (weeks)')
    # plt.ylabel('research units \n completed')
    plt.savefig(
        f'plots/vectors/fit_cluster_{cluster}.svg',
        format='svg', dpi=300)

    # basic
    plt.figure(figsize=(4, 4), dpi=300)
    params = result_fit[cluster, 1, 0]
    data = gen_data.gen_data_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, discount_factor=params[0], efficacy=params[1],
        effort_work=params[2], n_trials=n, thr=constants.THR,
        states_no=constants.STATES_NO)
    plot_trajectories(data, 'gray', 3, 1)
    plt.text(11, 0, f'$R^2$ = {np.round(metrics[cluster, 0, 0],2)}',
             fontsize=16)
    plt.savefig(
        f'plots/vectors/fit_basic_cluster_{cluster}.svg',
        format='svg', dpi=300)

    # efficacy gap
    plt.figure(figsize=(4, 4), dpi=300)
    params = result_fit[cluster, 1, 1]
    data = gen_data.gen_data_efficacy_gap(
        constants.STATES, constants.ACTIONS,  constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, discount_factor=params[0], efficacy_assumed=params[1],
        efficacy_actual=params[2], effort_work=params[3], n_trials=n,
        thr=constants.THR, states_no=constants.STATES_NO)
    plot_trajectories(data, 'gray', 3, 1)
    plt.text(11, 0, f'$R^2$ = {np.round(metrics[cluster, 1, 0],2)}',
             fontsize=16)
    plt.savefig(
        f'plots/vectors/fit_eff_gap_cluster_{cluster}.svg',
        format='svg', dpi=300)

    # convex concav
    plt.figure(figsize=(4, 4), dpi=300)
    params = result_fit[cluster, 1, 2]
    data = gen_data.gen_data_convex_concave(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, discount_factor=params[0], efficacy=params[3],
        effort_work=params[1], exponent=params[2], n_trials=n,
        thr=constants.THR, states_no=constants.STATES_NO)
    plot_trajectories(data, 'gray', 3, 1)
    plt.text(11, 0, f'$R^2$ = {np.round(metrics[cluster, 2, 0],2)}',
             fontsize=16)
    plt.savefig(
        f'plots/vectors/fit_conv_conc_cluster_{cluster}.svg',
        format='svg', dpi=300)

    # imm basic
    plt.figure(figsize=(4, 4), dpi=300)
    params = result_fit[cluster, 1, 3]
    data = gen_data.gen_data_immediate_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR, constants.REWARD_EXTRA, constants.REWARD_SHIRK,
        constants.BETA, discount_factor=params[0], efficacy=params[3],
        effort_work=params[1], exponent=params[2], n_trials=n,
        thr=constants.THR, states_no=constants.STATES_NO)
    plot_trajectories(data, 'gray', 3, 1)
    plt.text(11, 0, f'$R^2$ = {np.round(metrics[cluster, 3, 0],2)}',
             fontsize=16)
    plt.savefig(
        f'plots/vectors/fit_imm_basic_cluster_{cluster}.svg',
        format='svg', dpi=300)

    # different discount
    plt.figure(figsize=(4, 4), dpi=300)
    params = result_fit[cluster, 1, 4]
    data = gen_data.gen_data_diff_discounts(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_THR_DIFF_DISCOUNTS,
        constants.REWARD_EXTRA_DIFF_DISCOUNTS, constants.REWARD_SHIRK,
        constants.BETA_DIFF_DISCOUNTS, discount_factor_reward=params[0],
        discount_factor_cost=params[1], efficacy=params[2],
        effort_work=params[3], n_trials=n, thr=constants.THR,
        states_no=constants.STATES_NO)
    plot_trajectories(data, 'gray', 3, 1)
    plt.text(11, 0, f'$R^2$ = {np.round(metrics[cluster, 4, 0],2)}',
             fontsize=16)
    plt.savefig(
        f'plots/vectors/fit_diff_disc_cluster_{cluster}.svg',
        format='svg', dpi=300)

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
    plot_trajectories(data, 'gray', 3, 1)
    plt.text(11, 0, f'$R^2$ = {np.round(metrics[cluster, 5, 0],2)}',
             fontsize=16)
    plt.savefig(
        f'plots/vectors/fit_no_commit_cluster_{cluster}.svg',
        format='svg', dpi=300)

# %%
# calculate distance between parameters
for model in range(6):
    vectors = np.vstack(result_fit[:, 1, model])
    distance_matrix = get_distance_matrix(vectors)
    matrix = np.triu(distance_matrix)
    plt.figure(figsize=(5, 4), dpi=300)
    sns.heatmap(distance_matrix, cmap="crest", mask=matrix,
                vmin=0, vmax=41)
    plt.xlabel('cluster')
    plt.ylabel('cluster')
    # plt.savefig(
    #     f'plots/vectors/distance_matrix_model_{model}.svg',
    #     format='svg', dpi=300)
