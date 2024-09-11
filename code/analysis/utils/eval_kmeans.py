import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def evalKMeans(range_n_clusters, X, print_otp=False, metric='euclidean'):
    
    # Initialize results arrays
    silhouette_avg = np.empty((len(range_n_clusters)))
    clusters_labels = np.empty((len(range_n_clusters), X.shape[0]))

    # Iterate for each k
    for idx, n_clusters in enumerate(range_n_clusters):
        # Run clustering with n_clusters (random generator seed of 0 for reproducibility)
        clusterer = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(X)

        # Save clusters labels
        clusters_labels[idx,:] = cluster_labels
        clusters_labels = clusters_labels.astype(int)+1

        # Get the silhouette_score (average value for all the samples) as measure of goodness of clustering
        silhouette_avg[idx] = silhouette_score(X, cluster_labels, metric=metric)

        if print_otp:
            # Compute the silhouette score for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels, metric=metric)

            # Get number of samples with bad clustering 
            n_negatives = np.sum(sample_silhouette_values<0)

            # Get number of clusters with bad clustering (where max silouhette value is lower than the mean silohuette scores)
            max_cluster_silhouette_values = np.empty(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to cluster i, and get max
                max_cluster_silhouette_values[i] = np.max(sample_silhouette_values[cluster_labels == i])
                
            n_poor_clusts = np.sum(max_cluster_silhouette_values < silhouette_avg[idx])

            # Print
            print( "For n_clusters = :", n_clusters)
            print( "    The average silhouette_score is :", silhouette_avg[idx])
            print( "    The number of negative sample silhouette_score is :", n_negatives)
            print( "    The number of below average clusters is :", n_poor_clusts)
    
    return silhouette_avg, clusters_labels