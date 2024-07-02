import constants
import gen_data
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

# %%


def time_to_cross_thr(trajectories):

    times = []
    for trajectory in trajectories:

        if trajectory[-1] >= constants.THR:
            times.append(np.where(trajectory >= constants.THR)[0][0])
        else:
            times.append(np.nan)

    return times


# %%

# discounting delayed rewards

# delays with
efficacies = np.linspace(0.3, 1.0, 10)
discounts = [0.3, 0.6, 0.9, 0.98]

for discount_factor in discounts:

    delay_mn = []
    delay_sem = []
    for efficacy in efficacies:

        trajectories = gen_data.gen_data_basic(
            constants.STATES, constants.ACTIONS, constants.HORIZON,
            constants.REWARD_THR, constants.REWARD_EXTRA,
            constants.REWARD_SHIRK, constants.BETA, discount_factor, efficacy,
            constants.EFFORT_WORK, 1000, constants.THR, constants.STATES_NO)

        delays = time_to_cross_thr(trajectories)
        delay_mn.append(np.nanmean(delays))
        delay_sem.append(sem(delays, nan_policy='omit'))

    plt.errorbar(efficacies, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--')
sns.despine()
plt.ylabel('time to cross \n 14 units (in weeks)')
plt.xlabel('efficacy')


efforts = np.linspace(-1, -0.3, 10)
discounts = [0.3, 0.6, 0.9, 0.98]
plt.figure()
for discount_factor in discounts:

    delay_mn = []
    delay_sem = []
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

    plt.errorbar(efforts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--')
sns.despine()
plt.ylabel('time to cross \n 14 units (in weeks)')
plt.xlabel('effort')


# %%
# gap between real and assumed efficacies

discount_factor = 0.9
efficacys_assumed = np.linspace(0.2, 1, 10)
efficacys_real = [0.3, 0.6, 0.9]
plt.figure()
for efficacy_real in efficacys_real:

    delay_mn = []
    delay_sem = []
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

    plt.errorbar(efficacys_assumed, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--')
sns.despine()
plt.ylabel('time to cross \n 14 units (in weeks)')
plt.xlabel('efficacy_assumed')

# %%
# immediate rewrad cases

# nonlinearity in effort function

discounts = np.linspace(0.1, 1, 10)
nonlinearitys = [0.8, 1, 1.5, 2.2]
plt.figure()
for exponent in nonlinearitys:

    delay_mn = []
    delay_sem = []
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

    plt.errorbar(discounts, delay_mn, yerr=delay_sem, linewidth=3,
                 marker='o', linestyle='--')
sns.despine()
plt.ylabel('time to cross \n 14 units (in weeks)')
plt.xlabel('discount factor')

# %%
# different discount factors for effort and reward
