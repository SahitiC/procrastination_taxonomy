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


def time_to_finish(trajectories):
    """
    find when all units arre completed for each trajectory
    (of work) inputted; if threshold is never reached, returns NaN
    """

    times = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] == constants.STATES_NO-1:
            times.append(np.where(trajectory >= constants.STATES_NO-1)[0][0])
        else:
            times.append(np.nan)

    return times


def did_it_finish(trajectories):
    """
    find if all units have been completed for each trajectory inputted
    """

    completed = []
    trajectories = np.array(trajectories)

    for trajectory in trajectories[:, 1:]:

        if trajectory[-1] == constants.STATES_NO-1:
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
    plt.yticks([0, 10, 20])
    plt.ylim(-1, 21)
    plt.xlabel('time')
    plt.ylabel('Units of work \n completed')
    sns.despine()


# %% set seed, define color maps
cmap_blues = plt.get_cmap('Blues')
cmap_greens = plt.get_cmap('Greens')
cmap_RdPu = plt.get_cmap('RdPu')
cmap_oranges = plt.get_cmap('Oranges')
np.random.seed(0)

# %% discounting

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
            constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
            discount_factor, efficacy, constants.EFFORT_WORK, 1000,
            constants.STATES_NO)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(efficacies, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$\eta$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.09, 1.00, r'$\gamma$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/basic_efficacies_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\eta$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/basic_efficacies_rates.svg',
    format='svg', dpi=300)

# plot example trajectories
for i, d in enumerate(discounts[1:]):
    discount_factor, efficacy = d, constants.EFFICACY

    trajectories = gen_data.gen_data_basic(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
        discount_factor, efficacy, constants.EFFORT_WORK, 1000,
        constants.STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(fr'$\gamma$={np.round(d,2)}',
              fontsize=24)
    plt.savefig(
        f'plots/vectors/basic_efficacies_ex_{i}.svg',
        format='svg', dpi=300)
    plt.show()

# %% delays with efforts

efforts = np.linspace(-1.5, -0.3, 10)
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
            constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
            discount_factor, constants.EFFICACY, effort_work, 1000,
            constants.STATES_NO)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(efforts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel('$r_{effort}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.09, 1.00, r'$\gamma$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/basic_efforts_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$r_{effort}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/basic_efforts_rates.svg',
    format='svg', dpi=300)

# %% gap between real and assumed efficacies

discount_factor = 0.9
efficacys_assumed = np.linspace(0.3, 1, 10)
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
            constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
            discount_factor, efficacy_assumed, efficacy_real,
            constants.EFFORT_WORK, 1000, constants.STATES_NO)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(efficacys_assumed, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{efficacy_real}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n 14 complete task')
ax1.set_xlabel(r'$\eta_{assumed}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.12, 1.00, r'$\eta_{real}$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/eff_gap_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\eta_{assumed}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/eff_gap_rates.svg',
    format='svg', dpi=300)

for i, ef_a in enumerate([0.3, 0.6, 0.8]):

    efficacy_assumed, efficacy_real = ef_a, 0.9

    trajectories = gen_data.gen_data_efficacy_gap(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
        discount_factor, efficacy_assumed, efficacy_real,
        constants.EFFORT_WORK, 1000, constants.STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(fr'$\eta$={np.round(efficacy_assumed,2)}',
              fontsize=24)
    plt.savefig(
        f'plots/vectors/eff_gap_ex_{i}.svg',
        format='svg', dpi=300)
    plt.show()

# %% immediate reward cases
# nonlinearity in effort function

discounts = np.linspace(0.2, 1, 10)
nonlinearitys = [0.8, 1, 1.5, 2.2]
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
            constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
            discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
            exponent, 1000, constants.STATES_NO)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(discounts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{exponent}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$\gamma$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=5, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.08, 1.0, r'$k$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/conv_conc_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion rate', fontsize=20)
ax2.set_xlabel(r'$\gamma$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/conv_conc_rates.svg',
    format='svg', dpi=300)

# example trajectories
for i, e in enumerate([1.0, 1.2, 2.2]):
    discount_factor, exponent = discounts[-1], e

    trajectories = gen_data.gen_data_convex_concave(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_UNIT, constants.REWARD_SHIRK, constants.BETA,
        discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
        exponent, 1000, constants.STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(fr'$k$={np.round(exponent,2)}',
              fontsize=24)
    plt.savefig(
        f'plots/vectors/conv_conc_ex_{i}.svg',
        format='svg', dpi=300)
    plt.show()

# %% different discount factors for effort and reward

# plot example policy

reward_func = task_structure.reward_immediate(
    constants.STATES, constants.ACTIONS, constants.REWARD_SHIRK,
    constants.REWARD_UNIT_DIFF_DISCOUNTS)

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

discounts_reward = [0.5, 0.7, 0.8, 0.9]
discounts_cost = np.linspace(0.2, 1, 10)
cycle_colors = cycler('color',
                      cmap_oranges(np.linspace(0.3, 1, 4)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
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
            constants.REWARD_UNIT_DIFF_DISCOUNTS,
            constants.REWARD_SHIRK,
            constants.BETA_DIFF_DISCOUNTS, discount_factor_reward,
            discount_factor_cost, constants.EFFICACY, constants.EFFORT_WORK,
            1000, constants.STATES_NO)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(discounts_cost, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor_reward}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$\gamma_{c}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.08, 1.00, r'$\gamma_{r}$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/diff_disc_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel(r'$\gamma_{c}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/diff_disc_rates.svg',
    format='svg', dpi=300)

for i, d_c in enumerate([0.3, 0.5, 0.6, 0.75]):
    discount_factor_reward, discount_factor_cost = 0.9, d_c

    trajectories = gen_data.gen_data_diff_discounts(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        constants.REWARD_UNIT_DIFF_DISCOUNTS,
        constants.REWARD_SHIRK,
        constants.BETA_DIFF_DISCOUNTS, discount_factor_reward,
        discount_factor_cost, constants.EFFICACY, constants.EFFORT_WORK,
        1000, constants.STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(fr'$\gamma_c$={np.round(discount_factor_cost,2)}',
              fontsize=24)
    plt.savefig(
        f'plots/vectors/diff_disc_ex_{i}.svg',
        format='svg', dpi=300)
    plt.show()

# %% waiting for interesting rewards

rewards_interest = np.linspace(0.0, 4.5, 10)
discounts = [0.6, 0.95, 1]
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 3)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
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
            constants.HORIZON, constants.REWARD_UNIT, constants.REWARD_SHIRK,
            constants.BETA, constants.P_STAY_LOW, constants.P_STAY_HIGH,
            discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
            reward_interest, 1000,  constants.STATES_NO_NO_COMMIT)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(rewards_interest, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{discount_factor}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$r_{interest}$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.2), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.2, 0.95, r'$\gamma$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/no_commit_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel(r'$r_{interest}$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/no_commit_rates.svg',
    format='svg', dpi=300)

# example trajectories
for i, d in enumerate([0.95, 1.0]):
    reward_interest, discount_factor = rewards_interest[4], d

    trajectories = gen_data.gen_data_no_commitment(
        constants.STATES_NO_COMMIT, constants.ACTIONS_BASE,
        constants.HORIZON, constants.REWARD_UNIT, constants.REWARD_SHIRK,
        constants.BETA, constants.P_STAY_LOW, constants.P_STAY_HIGH,
        discount_factor, constants.EFFICACY, constants.EFFORT_WORK,
        reward_interest, 1000,  constants.STATES_NO_NO_COMMIT)

    plt.figure(figsize=(3, 3), dpi=300)
    plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(fr'$\gamma_r$={np.round(discount_factor,2)}',
              fontsize=24)
    plt.savefig(
        f'plots/vectors/no_commit_ex_{i}.svg',
        format='svg', dpi=300)
    plt.show()

# %% fatigue

discounts = np.linspace(0.5, 1, 10)
effort_highs = [-0.7, -1.5, -3]
cycle_colors = cycler('color',
                      cmap_blues(np.linspace(0.3, 1, 4)))

fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(4, 3), dpi=300)
ax1.set_prop_cycle(cycle_colors)
ax2.set_prop_cycle(cycle_colors)

plt.figure()

for effort_high in effort_highs:

    delay_mn = []
    delay_sem = []
    completion_rate = []
    for discount_factor in discounts:

        trajectories, _, _ = gen_data.gen_data_fatigue(
            constants.STATES_FATIGUE, constants.ACTIONS_BASE,
            constants.HORIZON, constants.REWARD_UNIT, constants.REWARD_SHIRK,
            constants.BETA, constants.P_LOW, constants.P_HIGH,
            discount_factor, constants.EFFICACY_FATIGUE,
            constants.EFFORT_LOW_FATIGUE, effort_high, 1000,
            constants.STATES_NO_FATIGUE)

        delays = time_to_finish(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))
        completions = did_it_finish(trajectories)
        completion_rate.append(np.nanmean(completions))

    ax1.errorbar(discounts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--', label=f'{effort_high}')

    ax2.plot(completion_rate, linewidth=3, marker='o', linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel(r'$\gamma$')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
fig1.text(0.05, 1.00, r'$r_{effort, high}$', ha='center', va='center')
fig1.savefig(
    'plots/vectors/fatigue_delays.svg',
    format='svg', dpi=300)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel(r'$\gamma$', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks([])
fig2.savefig(
    'plots/vectors/fatigue_rates.svg',
    format='svg', dpi=300)

effort_high, discount_factor = -3, 1.0

trajectories, trajectories_s, trajectories_actions = gen_data.gen_data_fatigue(
    constants.STATES_FATIGUE, constants.ACTIONS_BASE,
    constants.HORIZON, constants.REWARD_UNIT, constants.REWARD_SHIRK,
    constants.BETA, constants.P_LOW, constants.P_HIGH,
    discount_factor, constants.EFFICACY_FATIGUE,
    constants.EFFORT_LOW_FATIGUE, effort_high, 1000,
    constants.STATES_NO_FATIGUE)

plt.figure(figsize=(3, 3), dpi=300)
plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
plt.title(r'$\gamma_r$=1.0, $r_{effort, high}$=-3.0',
          fontsize=24)
plt.savefig(
    'plots/vectors/fatigue_ex_1.svg',
    format='svg', dpi=300)
plt.show()

trajectory = 9
plt.figure(figsize=(4, 3), dpi=300)
t = np.arange(constants.HORIZON)
fatigue = np.where(trajectories_s[trajectory]
                   > constants.STATES_NO_FATIGUE/2 - 1, 1, 0)
low_fatigue = fatigue[:-1] == 0
high_fatigue = fatigue[:-1] == 1
plt.plot(trajectories_actions[trajectory], linestyle='dashed', label='actions')
plt.plot(trajectories[trajectory], linestyle='dashed', label='units completed')
plt.scatter(t[low_fatigue], trajectories_actions[trajectory][low_fatigue],
            marker='s', facecolor='none', edgecolor='tab:blue',
            label='low fatigue')
plt.scatter(t[high_fatigue], trajectories_actions[trajectory][high_fatigue],
            marker='s', label='high fatigue')
sns.despine()
plt.xlabel('time')
plt.yticks([0, 10, 20])
plt.xticks([0, 7, 15])
plt.legend(bbox_to_anchor=(1.25, 0.75), ncol=1, frameon=False, fontsize=14,
           loc='upper center', columnspacing=0.5)
plt.savefig(
    'plots/vectors/fatigue_ex_actions.svg',
    format='svg', dpi=300)
plt.show()
