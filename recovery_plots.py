import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from functools import reduce
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['axes.linewidth'] = 2

# %% functions


def choose_best_model(fits, free_param_no):
    """
    description

    Parameters
    ----------
    fits : TYPE
        DESCRIPTION.
    free_param_no : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    likelihoods, params = fits[0], fits[1]
    model_recovered = np.argmin(
        2*np.array(likelihoods) + np.array(free_param_no) * 2)
    params_recovered = params[model_recovered]

    return [params_recovered, model_recovered]


def full_counts(target_numbers, unique_values, counts):

    # Create an array of zeros for all target numbers
    full_counts = np.zeros(len(target_numbers), dtype=int)

    # Find the indices in target_numbers where unique_values exist
    matching_indices = np.in1d(target_numbers, unique_values).nonzero()[0]

    # Place the counts in the corresponding positions in full_counts
    full_counts[matching_indices] = counts

    return full_counts


# %% load recovery results
result_recovery_fit_params = np.load('result_recovery_fit_params.npy',
                                     allow_pickle=True)
result_recovery = np.load('result_recovery.npy', allow_pickle=True)
input_recovery = np.load('inputs_recovery.npy', allow_pickle=True)
fit_params = np.load('fit_params.npy', allow_pickle=True)

# model names, params
models = ['basic', 'effic_gap', 'conv_conc', 'imm_basic',
          'diff_disc', 'no_commit']
model_no = [0, 1, 2, 3, 4, 5]
free_param_no = [3, 4, 4, 4, 4, 4]

# %% choose best fit model

# for randomly chosen parameters
models_recovered = []
for i in range(len(result_recovery)):

    models_recovered.append(
        choose_best_model(result_recovery[i, :, :], free_param_no))
models_recovered = np.array(models_recovered, dtype=object)

# for parameters returned from model fits
models_recovered_fit_params = []
for i in range(len(result_recovery_fit_params)):

    models_recovered_fit_params.append(
        choose_best_model(result_recovery_fit_params[i, :, :], free_param_no))
models_recovered_fit_params = np.array(models_recovered_fit_params,
                                       dtype=object)

# %% model recovery plots
# exclude ones where a recovered param =0 (while input params are not 0)
# ensure that 0 trajectories are excluded
index = []
for i in range(len(models_recovered)):
    if (np.any(models_recovered[i, 0] == 0) and
            np.all(input_recovery[i, 0] != 0)):
        index.append(i)
final_result = np.delete(models_recovered, index, axis=0)
final_inputs = np.delete(input_recovery, index, axis=0)

# plot counts of returned models for each input model type
freq_recovered = []
freq_recovered_fit_params = []
for i in range(len(models)):

    # for randomly chosen params
    index = np.where(final_inputs[:, 1] == i)[0]
    frequency = np.unique(final_result[index, 1], return_counts=True)
    freq_recovered.append(full_counts(model_no, frequency[0], frequency[1]))

    # for params from model fits
    index_fit_params = np.where(fit_params[:, 1] == i)[0]
    frequency_fit_params = np.unique(
        models_recovered_fit_params[index_fit_params, 1], return_counts=True)
    freq_recovered_fit_params.append(
        full_counts(model_no, frequency_fit_params[0],
                    frequency_fit_params[1]))

plt.figure(figsize=(5, 4), dpi=300)
sns.heatmap(freq_recovered, cmap='vlag')
plt.xlabel('model')
plt.yticks([])

plt.figure(figsize=(5, 4), dpi=300)
sns.heatmap(freq_recovered_fit_params, cmap='vlag')
plt.xlabel('model')
plt.yticks([])

# %% parameter recovery lots
