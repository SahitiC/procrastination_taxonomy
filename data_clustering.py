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

                plt.plot(ast.literal_eval(data[i]), alpha=0.5)

        sns.despine()
        plt.xticks([0, 7, 15])
        plt.yticks([0, 5, 11])


def silhoutte_plots(data, labels, n_clusters, **kwargs):
    """
    plot silhouette score for each sample (in a sorted order) given data
    and labels
    (code adapted from sklearn example:
     https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
    """

    sample_silhouette_values = silhouette_samples(timeseries_to_cluster,
                                                  labels)
    y_lower = 10
    plt.figure()
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
        plt.ylim([0, len(timeseries_to_cluster) + (n_clusters + 1) * 10])
        # Label the silhouette plots with their cluster numbers at the middle
        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # slihoutte score
    silhouette_avg = silhouette_score(timeseries_to_cluster, labels)
    # The vertical line for average silhouette score of all the values
    plt.axvline(x=silhouette_avg, color="red", linestyle="--")
    plt.yticks([])
    plt.ylabel('cluster label')
    plt.xlabel('silhouette value')


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
    silhouette_scores = []
    for cluster_size in range(1, 15):
        print(cluster_size+1)
        km = KMeans(n_clusters=cluster_size+1, n_init=10,
                    random_state=0)
        labels = km.fit_predict(timeseries_to_cluster)
        inertia.append(km.inertia_)
        silhouette_scores.append(silhouette_score(timeseries_to_cluster,
                                                  labels))
        silhoutte_plots(timeseries_to_cluster, labels, cluster_size+1)

    plt.figure(figsize=(7, 6))
    plt.plot(inertia)
    plt.xticks(np.arange(14), labels=np.arange(2, 16))
    plt.xlabel('cluster number')
    plt.ylabel('k-means sum of squares')

    plt.figure(figsize=(7, 6))
    plt.plot(silhouette_scores)
    plt.xticks(np.arange(14), labels=np.arange(2, 16))
    plt.xlabel('cluster number')
    plt.ylabel('silhouette score')

    # final k means clustering using the best cluster size
    # best cluster size = around the elbow in inertia plot (around 6, 7, 8)
    # choose one with similar cluster sizes and silhoutte scores across
    # samples and clusters (cluster_size=8)
    km = KMeans(n_clusters=8, n_init=10, random_state=0, verbose=True)
    labels = km.fit_predict(timeseries_to_cluster)
    data_relevant['labels'] = labels

    # plot clustered data
    plot_clustered_data(data_relevant['cumulative progress weeks'],
                        data_relevant['labels'])

    data_relevant.to_csv('data/data_clustered.csv', index=False)
