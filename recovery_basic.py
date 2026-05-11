# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

# %%
result_recovery = np.load('result_recovery.npy', allow_pickle=True)
input_recovery = np.load('inputs_recovery.npy', allow_pickle=True)

free_param_model = 3
input_params = []
for i in range(len(input_recovery)):

    input_params.append(input_recovery[i, 0][-3-free_param_model:-3])

result_params = np.vstack(np.hstack(result_recovery[:, 1, :]))
input_params = np.array(input_params)

# %%
index = []
for i in range(len(input_params)):
    if (np.any(result_params[i, :] == 0) and
            np.all(input_params[i, :] != 0)):
        index.append(i)
final_result = np.delete(result_params, index, axis=0)
final_inputs = np.delete(input_params, index, axis=0)

# %%
lim = [(-0.05, 1.05), (-0.05, 1.05), (-3, 0.05)]

for i in range(free_param_model):
    plt.figure(figsize=(5, 5))
    corr = np.corrcoef(final_inputs[:, i], final_result[:, i])[0, 1]
    plt.scatter(final_inputs[:, i], final_result[:, i])
    plt.plot(
        np.linspace(lim[i][0], lim[i][1], 10),
        np.linspace(lim[i][0], lim[i][1], 10),
        linewidth=1, color='black')  # x=y
    plt.title(f'corr = {np.round(corr, 2)}')
    sns.despine()
