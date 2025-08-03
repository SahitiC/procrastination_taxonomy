
rewards_unit = [0.5, 1, 1.5, 2, 2.5]


fig1, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)

plt.figure()

delay_mn = []
delay_sem = []
completion_rate = []
for reward_unit in rewards_unit:

    trajectories = gen_data.gen_data_diff_discounts(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        reward_unit,
        constants.REWARD_SHIRK,
        constants.BETA_DIFF_DISCOUNTS, 0.9, 0.6,
        constants.EFFICACY, constants.EFFORT_WORK,
        1000, constants.STATES_NO)

    delays = time_to_finish(trajectories)
    delay_mn.append(np.nanmean(delays))
    delay_sem.append(sem(delays, nan_policy='omit'))
    completions = did_it_finish(trajectories)
    completion_rate.append(np.nanmean(completions))

ax1.errorbar(rewards_unit, delay_mn, yerr=delay_sem, linewidth=3,
             marker='o', linestyle='--')

ax2.plot(rewards_unit, completion_rate, linewidth=3, marker='o',
         linestyle='--')

sns.despine(ax=ax1)
ax1.set_ylabel('Avg. time to \n complete task')
ax1.set_xlabel('reward for completion')
ax1.set_yticks([0, 5, 10, 15])
ax1.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False, fontsize=18,
           loc='upper center', columnspacing=0.5)
ax1.set_title(r'$\gamma_r$ = 0.9, $\gamma_c$=0.6')
ax1.set_xticks(rewards_unit)

sns.despine(ax=ax2)
ax2.set_ylabel('Completion \n rate', fontsize=20)
ax2.set_xlabel('reward for completion', fontsize=20)
ax2.set_yticks([0, 1])
ax2.set_xticks(rewards_unit)

for reward_unit in rewards_unit:

    trajectories = gen_data.gen_data_diff_discounts(
        constants.STATES, constants.ACTIONS, constants.HORIZON,
        reward_unit,
        constants.REWARD_SHIRK,
        constants.BETA_DIFF_DISCOUNTS, 0.9, 0.6,
        constants.EFFICACY, constants.EFFORT_WORK,
        1000, constants.STATES_NO)

    plt.figure(figsize=(3, 3), dpi=300)
    plot_trajectories(trajectories, 'black', 2, 0.5, number_samples=10)
    plt.title(f'reward={np.round(reward_unit,2)}',
              fontsize=24)
    plt.show()
