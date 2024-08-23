import seaborn as sns
import numpy as np
import pandas as pd
import ast
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 3

# %%

data = pd.read_csv('data/zhang_ma_data.csv')

# drop NaN entries
data_relevant = data.dropna(subset=['delta progress'])
data_relevant = data_relevant.reset_index(drop=True)

semester_length = len(ast.literal_eval(data_relevant['delta progress'][0]))

# who completed lesser than 7 hrs, when did they do their credits
participants_less7 = data_relevant[data_relevant['Total credits'] < 7]
participants_less7 = participants_less7.reset_index(drop=True)
plt.figure(figsize=(6, 5))
for i in range(len(participants_less7)):
    plt.plot(np.arange(semester_length),
             ast.literal_eval(participants_less7['cumulative progress'][i]),
             linewidth=2,
             label=f" {participants_less7['SUB_INDEX_194'][i]} ")
plt.legend(frameon=False, title='sub index')
plt.ylabel('cumulative hours completed')
plt.xlabel('day in semester')

# drop the ones that discontinued (subj. 1, 95, 111)
# they report to have discountinued in verbal response in 'way_allocate' column
# they do 1 hour in the very beginning and then nothing after
data_relevant = data_relevant.drop([1, 90, 104])
data_relevant = data_relevant.reset_index(drop=True)

# drop ones who complete more than 11 hours
# as extra credit ends at 11 hours
mask = np.where(data_relevant['Total credits'] <= 11)[0]
data_relevant = data_relevant.loc[mask]
data_relevant['Total credits']
data_relevant = data_relevant.reset_index(drop=True)

# %%

# normalise cumulative series
cumulative_normalised = []
for i in range(len(data_relevant)):
    temp = ast.literal_eval(data_relevant['cumulative progress'][i])
    cumulative_normalised.append(list(temp/data_relevant['Total credits'][i]))
data_relevant['cumulative progress normalised'] = cumulative_normalised

# delta progress week wise
semester_length_weeks = round(semester_length/7)
delta_progress_weeks = []
for i in range(len(data_relevant)):

    temp = ast.literal_eval(data_relevant['delta progress'][i])
    temp_week = []
    for i_week in range(semester_length_weeks):

        temp_week.append(
            sum(temp[i_week*7: (i_week+1)*7]) * 1.0)

    assert sum(temp_week) == data_relevant['Total credits'][i]
    delta_progress_weeks.append(temp_week)
    plt.plot(temp_week)

data_relevant['delta progress weeks'] = delta_progress_weeks

# cumulative progress week wise
cumulative_progress_weeks = []
plt.figure()
for i in range(len(data_relevant)):

    cumulative_progress_weeks.append(
        list(np.cumsum(data_relevant['delta progress weeks'][i])))
    plt.plot(
        np.cumsum(data_relevant['delta progress weeks'][i]))

data_relevant['cumulative progress weeks'] = cumulative_progress_weeks

# choose completion trajectories to save
data_subset = data_relevant[['SUB_INDEX_194', 'Total credits',
                             'delta progress', 'cumulative progress',
                             'cumulative progress normalised',
                             'delta progress weeks',
                             'cumulative progress weeks']]

data_subset.to_csv('data/data_preprocessed.csv', index=False)
