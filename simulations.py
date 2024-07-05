"""
simulations of the five models (reproducing Figs 2-4)
"""

import constants
import gen_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

# %%


def time_to_cross_thr(trajectories):
    """
    find when threshold number of units is first reached for each trajectory
    (of work) inputted; if threshold is never reached, returns NaN

    params:
        trajectories (list): list of trajectories

    returns:
        times (list): list of timesteps when threshold is first reached
    """

    times = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] >= constants.THR:
            times.append(np.where(trajectory >= constants.THR)[0][0])
        else:
            times.append(np.nan)

    return times


def did_it_cross_thr(trajectories):
    """
    find if threshold number of units reached for each trajectory inputted

    params:
        trajectories (list): list of trajectories

    returns:
        times (list): whether each trajectory reached threshol (1) or not (0)
    """

    completed = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] >= constants.THR:
            completed.append(1)
        else:
            completed.append(0)

    return completed


# %%
cmap_blues = plt.get_cmap('Blues')
cmap_greens = plt.get_cmap('Greens')
cmap_RdPu = plt.get_cmap('RdPu')
cmap_oranges = plt.get_cmap('Oranges')

# %%
# discounting delayed rewards

# delays and completion rates with efficacy
efficacies = np.linspace(0.3, 0.98, 10)
discounts = [0.3, 0.6, 0.95, 1]
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 4)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

for discount_factor in discounts:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for efficacy in efficacies:

        trajectories = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
            constants.EFFORT_WORK, 1000, constants.THR, constants.STATES_NO)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_cross_thr(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(efficacies, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to cross \n 14 units (in weeks)')
ax1.set_xlabel('efficacy')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel('efficacy', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])

# example trajectories

# %%
# delays with efforts
efforts = np.linspace(-1, -0.3, 10)
discounts = [0.3, 0.6, 0.95, 1]
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 4)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

for discount_factor in discounts:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for effort_work in efforts:

        trajectories = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor,
            constants.EFFICACY, effort_work, 1000, constants.THR,
            constants.STATES_NO)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_cross_thr(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(efforts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to cross \n 14 units (in weeks)')
ax1.set_xlabel('efforts')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel('efforts', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])


# %%
# gap between real and assumed efficacies

discount_factor = 0.9
efficacys_assumed = np.linspace(0.2, 1, 10)
efficacys_real = [0.3, 0.6, 0.9]
cycle_colors = cycler('color',
                      cmap_greens(np.linspace(0.3, 1, 3)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

for efficacy_real in efficacys_real:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for efficacy_assumed in efficacys_assumed:

        trajectories = gen_data.gen_data_efficacy_gap(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA,
            discount_factor, efficacy_assumed, efficacy_real,
            constants.EFFORT_WORK, 1000, constants.THR, constants.STATES_NO)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_cross_thr(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(efficacys_assumed, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{efficacy_real}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to cross \n 14 units (in weeks)')
ax1.set_xlabel('efficacy assumed')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel('efficacy assumed', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])

# %%
# immediate reward cases

# nonlinearity in effort function

discounts = np.linspace(0.2, 1, 10)
nonlinearitys = [0.8, 1, 1.5, 2.2]
cycle_colors = cycler('color',
                      cmap_RdPu(np.linspace(0.3, 1, 4)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

plt.figure()
for exponent in nonlinearitys:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for discount_factor in discounts:

        trajectories = gen_data.gen_data_immediate_basic(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA,
            discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
            exponent, 1000, constants.THR, constants.STATES_NO)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_cross_thr(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(discounts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{exponent}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to cross \n 14 units (in weeks)')
ax1.set_xlabel('discount factors')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel('discount factors', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])

# %%
# different discount factors for effort and reward

discounts_reward = [0.5, 0.7, 0.8, 0.9]
discounts_cost = np.linspace(0.2, 1, 10)
cycle_colors = cycler('color',
                      cmap_oranges(np.linspace(0.3, 1, 4)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 2), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

plt.figure()
for discount_factor_reward in discounts_reward:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for discount_factor_cost in discounts_cost:

        trajectories = gen_data.gen_data_diff_discounts(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR_DIFF_DISCOUNTS,
            constants.REWARD_EXTRA_DIFF_DISCOUNTS, constants.REWARD_SHIRK,
            constants.BETA_DIFF_DISCOUNTS, discount_factor_reward,
            discount_factor_cost, constants.EFFICACY, constants.EFFORT_WORK,
            1000, constants.THR, constants.STATES_NO)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(discounts_cost, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor_reward}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to cross \n 14 units (in weeks)')
ax1.set_xlabel('discount factor cost')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel('discount factor cost', fontsize=20)
ax2.set_yticks([1])
ax2.set_xticks([])

# %%
# waiting for interesting rewards

rewards_interest = np.linspace(0.0, 6, 10)
discounts = [0.6, 0.95, 1]
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 3)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 2), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

plt.figure()
for discount_factor in discounts:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for reward_interest in rewards_interest:

        trajectories = gen_data.gen_data_no_commitment(
            constants.STATES_NO_COMMIT, constants.ACTIONS_BASE,
            constants.HORIZON, constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, constants.P_STAY_LOW,
            constants.P_STAY_HIGH, discount_factor, constants.EFFICACY,
            constants.EFFORT_WORK, reward_interest, 1000, constants.THR,
            constants.STATES_NO_NO_COMMIT)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(rewards_interest, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to cross \n 14 units (in weeks)')
ax1.set_xlabel('reward interest')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel('reward interest', fontsize=20)
ax2.set_yticks([1])
ax2.set_xticks([])
