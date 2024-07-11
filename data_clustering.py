from sklearn.metrics import silhouette_samples, silhouette_score
import numpy as np
import pandas as pd
import ast
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
mpl.rcParams['font.size'] = 24
mpl.rcParams['lines.linewidth'] = 3


# %%


def plot_clustered_data(data, labels, **kwargs):
    """
    plot trajectories in each cluster given data and labels
    """

    for label in set(labels):
        plt.figure(figsize=(4, 4), dpi=100)

        for i in range(len(data)):

            if labels[i] == label:

                plt.plot(np.array(ast.literal_eval(data[i])) * 2, alpha=0.5)

        sns.despine()
        plt.xticks([0, 7, 15])
        plt.yticks([0, 11, 14, 22])
        plt.xlabel('time (weeks)')
        plt.ylabel('research units \n completed')

        plt.savefig(
            f'plots/vectors/cluster_{label}.svg',
            format='svg', dpi=300)


# %%

if __name__ == "__main__":

    data_relevant = pd.read_csv('data/data_preprocessed.csv')

    # cluster normalised trajectories
    timeseries_to_cluster = []
    for i in range(len(data_relevant)):
        timeseries_to_cluster.append((ast.literal_eval(
            data_relevant['cumulative progress normalised'][i])))

    timeseries_to_cluster = np.array(timeseries_to_cluster)

    # cluster trajectories using k means for a range of cluster numbers: 2-15
    # plot inertia vs cluster number
    # plot silhoutte score for each cluster number
    inertia = []
    for cluster_size in range(1, 15):
        print(cluster_size+1)
        km = KMeans(n_clusters=cluster_size+1, n_init=10,
                    random_state=0)
        labels = km.fit_predict(timeseries_to_cluster)
        inertia.append(km.inertia_)

    plt.figure(figsize=(7, 6))
    plt.plot(inertia)
    plt.xticks(np.arange(14), labels=np.arange(2, 16))
    plt.xlabel('cluster number')
    plt.ylabel('k-means sum of squares')

    # final k means clustering using the best cluster size
    # best cluster size = around the elbow in inertia plot (around 6, 7, 8)
    km = KMeans(n_clusters=8, n_init=10, random_state=0, verbose=True)
    labels = km.fit_predict(timeseries_to_cluster)
    data_relevant['labels'] = labels

    # plot clustered data
    plot_clustered_data(data_relevant['cumulative progress weeks'],
                        data_relevant['labels'])

    data_relevant.to_csv('data/data_clustered.csv', index=False)
