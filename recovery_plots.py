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

# parameters of each model
model_params = [
    ['$\gamma$', '$\eta$', '$r_{effort}$'],
    ['$\gamma$', '$\eta_{assumed}$', '$\eta_{real}$', '$r_{effort}$'],
    ['$\gamma$', '$\eta$', '$r_{effort}$', 'k'],
    ['$\gamma$', '$\eta$', '$r_{effort}$', 'k'],
    ['$\gamma_{reward}$', '$\gamma_{effort}$', '$\eta$', '$r_{effort}$'],
    ['$\gamma$', '$\eta$', '$r_{effort}$', '$r_{interest}$']]

# parameter limits for each model
param_lim = [
    [(-0.05, 1.05), (-0.05, 1.05), (-3, 0.05)],
    [(-0.05, 1.05), (-0.05, 1.05), (-0.05, 1.05), (-3, 0.05)],
    [(-0.05, 1.05), (-0.05, 1.05), (-3, 0.05), (-0.05, 1.55)],
    [(-0.05, 1.05), (-0.05, 1.05), (-3.2, 0.05), (-0.05, 1.55)],
    [(-0.05, 1.05), (-0.05, 1.05), (0, 1.05), (-1, 0.05)],
    [(-0.05, 1.05), (-0.05, 1.05), (-1.65, 0), (-0.05, 15)]]

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
result_recovery_trimmed = np.delete(result_recovery, index, axis=0)

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

plt.figure(figsize=(6, 5), dpi=300)
sns.heatmap(freq_recovered, cmap='vlag')
plt.xlabel('model recovered')
plt.ylabel('model input')
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
           ['basic', 'eff-gap', 'conv-conc', 'imm-basic', 'diff-disc',
            'no-commit'], rotation=0, fontsize=16)
plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
           ['basic', 'eff-gap', 'conv-conc', 'imm-basic', 'diff-disc',
            'no-commit'], rotation=70, fontsize=16)
plt.savefig(
    'plots/vectors/recovery_model.svg',
    format='svg', dpi=300)

plt.figure(figsize=(6, 5), dpi=300)
sns.heatmap(freq_recovered_fit_params, cmap='vlag')
plt.xlabel('model recovered')
plt.ylabel('model input')
plt.yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
           ['basic', 'eff-gap', 'conv-conc', 'imm-basic', 'diff-disc',
            'no-commit'], rotation=0, fontsize=16)
plt.xticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
           ['basic', 'eff-gap', 'conv-conc', 'imm-basic', 'diff-disc',
            'no-commit'], rotation=70, fontsize=16)
plt.savefig(
    f'plots/vectors/recovery_model_fits.svg',
    format='svg', dpi=300)

# %% extract free parameters from input parameters
params = []
for model in range(len(models)):

    temp = final_inputs[np.where(final_inputs[:, 1] == model), 0].T

    for i in range(len(temp)):

        params.append(temp[i, 0][-3-free_param_no[model]:-3])

final_inputs[:, 0] = np.array(params, dtype=object)

# %% parameter recovery plots
# regardless of whether model as recovered, what were the params recovered from
# fitting the correct model


for model in range(len(models)):

    # choose input fitted params correspnding to model = model
    input_params_fit = fit_params[np.where(fit_params[:, 1] == model), 0]
    input_params_fit = np.vstack(np.hstack(input_params_fit))

    # choose the params recovered when fitting the same model
    result_params_fit = result_recovery_fit_params[
        np.where(fit_params[:, 1] == model), 1, model]
    result_params_fit = np.vstack(np.hstack(result_params_fit))

    # choose input fitted params correspnding to model = model
    input_params = final_inputs[np.where(final_inputs[:, 1] == model), 0]
    input_params = np.vstack(np.hstack(input_params))

    # choose the params recovered when fitting the same model
    result_params = result_recovery_trimmed[
        np.where(final_inputs[:, 1] == model), 1, model]
    result_params = np.vstack(np.hstack(result_params))

    for param in range(len(model_params[model])):

        plt.figure(figsize=(4, 4), dpi=300)

        plt.scatter(input_params[:, param], result_params[:, param])
        plt.scatter(input_params_fit[:, param], result_params_fit[:, param],
                    marker='x')

        lim = param_lim[model][param]

        plt.plot(
            np.linspace(lim[0], lim[1], 10),
            np.linspace(lim[0], lim[1], 10),
            linewidth=1, color='black')  # x=y line

        plt.xlim(lim)
        plt.ylim(lim)

        plt.xlabel(fr'true {model_params[model][param]}')
        plt.ylabel(fr'estimated {model_params[model][param]}')

        sns.despine()

        plt.savefig(
            f'plots/vectors/recover_params_{model}_{param}.svg',
            format='svg', dpi=300)
