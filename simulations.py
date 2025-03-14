"""
simulations of the five models 
Reproduces Figs 4-5
"""

import constants
import gen_data
import task_structure
import mdp_algms
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

# %% functions


def time_to_cross_thr(trajectories):
    """
    find when threshold number of units is first reached for each trajectory
    (of work) inputted; if threshold is never reached, returns NaN
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
    """

    completed = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] >= constants.THR:
            completed.append(1)
        else:
            completed.append(0)

    return completed


def plot_trajectories(trajectories, color, lwidth_mean, lwidth_sample,
                      number_samples):
    """
    plot input trajectories 
    """
    mean = np.mean(trajectories, axis=0)

    plt.plot(mean, color=color, linewidth=lwidth_mean)
    for i in range(number_samples):
        plt.plot(trajectories[i], color=color,
                 linewidth=lwidth_sample, linestyle='dashed')
    plt.xticks([0, 7, 15])
    plt.yticks([0, 11, 22])
    plt.ylim(-1, 22)
    plt.xlabel('time (weeks)')
    plt.ylabel('Research units \n completed')
    sns.despine()


# %% set seed, define color maps
cmap_blues = plt.get_cmap('Blues')
cmap_greens = plt.get_cmap('Greens')
cmap_RdPu = plt.get_cmap('RdPu')
cmap_oranges = plt.get_cmap('Oranges')
np.random.seed(0)

# %% discounting delayed rewards

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
ax1.set_xlabel(r'$\eta$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.09, 0.95, r'$\gamma$', ha='center', va='center')

fig1.savefig(
    'plots/vectors/basic_efficacies_delays.svg',
    format='svg', dpi=300)
fig1.show()


sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\eta$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])

fig2.savefig(
    'plots/vectors/basic_efficacies_rates.svg',
    format='svg', dpi=300)
fig2.show()

# plot example trajectories
discount_factor, efficacy = 1, efficacies[-2]

trajectories = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
    constants.EFFORT_WORK, 1000, constants.THR, constants.STATES_NO)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma$={np.round(discount_factor,2)}, $\eta$={np.round(efficacy,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/basic_efficacies_ex1.svg',
    format='svg', dpi=300)
plt.show()

discount_factor, efficacy = 0.3, efficacies[-3]

trajectories = gen_data.gen_data_basic(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
    constants.EFFORT_WORK, 1000, constants.THR, constants.STATES_NO)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma$={np.round(discount_factor,2)}, $\eta$={np.round(efficacy,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/basic_efficacies_ex2.svg',
    format='svg', dpi=300)
plt.show()

# %% delays with efforts

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
ax1.set_xlabel('$r_{effort}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.09, 0.95, r'$\gamma$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/basic_efforts_delays.svg',
    format='svg', dpi=300)
fig1.show()

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel('$r_{effort}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/basic_efforts_rates.svg',
    format='svg', dpi=300)
fig2.show()

# %% gap between real and assumed efficacies

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
ax1.set_xlabel(r'$\eta_{assumed}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.12, 0.95, r'$\eta_{real}$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/eff_gap_delays.svg',
    format='svg', dpi=300)
fig1.show()

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\eta_{assumed}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/eff_gap_rates.svg',
    format='svg', dpi=300)
fig2.show()

# example trajectories
efficacy_assumed, efficacy_real = efficacys_assumed[1], 0.9

trajectories = gen_data.gen_data_efficacy_gap(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA,
    discount_factor, efficacy_assumed, efficacy_real,
    constants.EFFORT_WORK, 1000, constants.THR, constants.STATES_NO)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma$={np.round(discount_factor,2)}, $\eta$={np.round(efficacy_assumed,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/eff_gap_ex1.svg',
    format='svg', dpi=300)
plt.show()

# %% nonlinearity in effort function (with delayed rewards)

discounts = np.linspace(0.2, 1, 10)
nonlinearitys = [0.5, 0.8, 1, 1.5, 2.2]
cycle_colors = cycler('color',
                      cmap_RdPu(np.linspace(0.3, 1, 5)))

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

        trajectories = gen_data.gen_data_convex_concave(
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
ax1.set_xlabel(r'$\gamma$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=5, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.03, 0.95, r'$k$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/conv_conc_delays.svg',
    format='svg', dpi=300)
fig1.show()

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\gamma$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/conv_conc_rates.svg',
    format='svg', dpi=300)
fig2.show()

# example trajectories
discount_factor, exponent = discounts[-1], 1.5

trajectories = gen_data.gen_data_convex_concave(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA,
    discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
    exponent, 1000, constants.THR, constants.STATES_NO)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma$={np.round(discount_factor,2)}, $k$={np.round(exponent,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/conv_conc_ex1.svg',
    format='svg', dpi=300)
plt.show()

# %% immediate reward cases

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
ax1.set_xlabel(r'$\gamma$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.09, 0.95, r'$k$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/imm_basic_delays.svg',
    format='svg', dpi=300)
fig1.show()

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\gamma$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/imm_basic_rates.svg',
    format='svg', dpi=300)
fig2.show()

# example trajectories
discount_factor, exponent = discounts[4], 2.2

trajectories = gen_data.gen_data_immediate_basic(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA,
    discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
    exponent, 1000, constants.THR, constants.STATES_NO)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma$={np.round(discount_factor,2)}, $k$={np.round(exponent,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/imm_basic_ex1.svg',
    format='svg', dpi=300)
plt.show()

# %% different discount factors for effort and reward

# plot example policy

reward_func = task_structure.reward_threshold(
    constants.STATES, constants.ACTIONS, constants.REWARD_SHIRK,
    constants.REWARD_THR_DIFF_DISCOUNTS, constants.REWARD_EXTRA_DIFF_DISCOUNTS,
    constants.THR, constants.STATES_NO)

effort_func = task_structure.effort(
    constants.STATES, constants.ACTIONS, constants.EFFORT_WORK)

reward_func_last = np.zeros(len(constants.STATES))
effort_func_last = np.zeros(len(constants.STATES))
T = task_structure.T_binomial(constants.STATES, constants.ACTIONS,
                              constants.EFFICACY)

V_opt_full, policy_opt_full, Q_values_full = (
    mdp_algms.find_optimal_policy_diff_discount_factors(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        0.9, 0.5,
        reward_func, effort_func, reward_func_last, effort_func_last, T)
)

policy_init_state = [policy_opt_full[i][0] for i in range(constants.HORIZON)]
policy_init_state = np.array(policy_init_state)
fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
cmap = mpl.colormaps['winter']
sns.heatmap(policy_init_state, linewidths=.5, cmap=cmap, cbar=True)
ax.set_xlabel('timestep')
ax.set_ylabel('horizon')
ax.tick_params(axis='x', labelrotation=90)
colorbar = ax.collections[0].colorbar
colorbar.set_label('actions:\n no. of units', rotation=270, labelpad=45)
fig.savefig(
    'plots/vectors/diff_disc_policy.svg',
    format='svg', dpi=300)
fig.show()

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
ax1.set_xlabel(r'$\gamma_{c}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.08, 0.95, r'$\gamma_{r}$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/diff_disc_delays.svg',
    format='svg', dpi=300)
fig1.show()

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel(r'$\gamma_{c}$', fontsize=20)
ax2.set_yticks([1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/diff_disc_rates.svg',
    format='svg', dpi=300)
fig2.show()

# example trajectories
discount_factor_reward, discount_factor_cost = 0.9, discounts_cost[3]

trajectories = gen_data.gen_data_diff_discounts(
    constants.STATES, constants.ACTIONS, constants.HORIZON,
    constants.REWARD_THR_DIFF_DISCOUNTS,
    constants.REWARD_EXTRA_DIFF_DISCOUNTS, constants.REWARD_SHIRK,
    constants.BETA_DIFF_DISCOUNTS, discount_factor_reward,
    discount_factor_cost, constants.EFFICACY, constants.EFFORT_WORK,
    1000, constants.THR, constants.STATES_NO)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma_r$={np.round(discount_factor_reward,2)}, $\gamma_c$={np.round(discount_factor_cost,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/diff_disc_ex1.svg',
    format='svg', dpi=300)
plt.show()

# %% waiting for interesting rewards

rewards_interest = np.linspace(0.0, 5, 10)
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
ax1.set_xlabel(r'$r_{interest}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.2, 0.95, r'$\gamma$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/no_commit_delays.svg',
    format='svg', dpi=300)
fig1.show()

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel(r'$r_{interest}$', fontsize=20)
ax2.set_yticks([1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/no_commit_rates.svg',
    format='svg', dpi=300)
fig2.show()

# example trajectories
reward_interest, discount_factor = rewards_interest[4], 0.95

trajectories = gen_data.gen_data_no_commitment(
    constants.STATES_NO_COMMIT, constants.ACTIONS_BASE,
    constants.HORIZON, constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, constants.P_STAY_LOW,
    constants.P_STAY_HIGH, discount_factor, constants.EFFICACY,
    constants.EFFORT_WORK, reward_interest, 1000, constants.THR,
    constants.STATES_NO_NO_COMMIT)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma_r$={np.round(discount_factor,2)}, $r$={np.round(reward_interest,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/no_commit_ex1.svg',
    format='svg', dpi=300)
plt.show()

reward_interest, discount_factor = rewards_interest[4], 1

trajectories = gen_data.gen_data_no_commitment(
    constants.STATES_NO_COMMIT, constants.ACTIONS_BASE,
    constants.HORIZON, constants.REWARD_THR, constants.REWARD_EXTRA,
    constants.REWARD_SHIRK, constants.BETA, constants.P_STAY_LOW,
    constants.P_STAY_HIGH, discount_factor, constants.EFFICACY,
    constants.EFFORT_WORK, reward_interest, 1000, constants.THR,
    constants.STATES_NO_NO_COMMIT)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(fr'$\gamma_r$={np.round(discount_factor,2)}, $r$={np.round(reward_interest,2)}',
          fontsize=24)
plt.savefig(
    'plots/vectors/no_commit_ex2.svg',
    format='svg', dpi=300)
plt.show()
