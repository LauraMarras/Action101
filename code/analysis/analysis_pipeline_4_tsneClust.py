import os
import numpy as np
from nilearn import image
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from utils.eval_kmeans import evalKMeans
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def clustering(results_group, n_clust_max, atlas_file):

    silhouette_avg, clusters_labels = evalKMeans(range(2, n_clust_max+1), results_group, print_otp=False)
    clusters_labels += 1

    # Load Atlas
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    x,y,z = atlas.get_fdata().shape
        
    # Initialize volume
    image_final = np.zeros((x, y, z, n_clust_max+1-2))

    # Create volume with cluster values for each k
    for k in range(n_clust_max+1-2):
        for r in range(results_group.shape[0]):
            x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==r+1)
            image_final[x_inds, y_inds, z_inds, k] = clusters_labels[k,r]
        
    # Create nifti
    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename('/data1/Action_teresi/CCA/{}_clustering.nii.gz'.format(atlas_file))

    return silhouette_avg, clusters_labels

def save_nifti(results, briks, atlas_file='Schaefer200', savepath='', ROIsmask=[]):
    
    atlas = image.load_img('/data1/Action_teresi/CCA/atlas/Schaefer_7N_{}.nii.gz'.format(atlas_file[-3:]))
    atlas_rois = np.unique(atlas.get_fdata()).astype(int)
    atlas_rois = np.delete(atlas_rois, np.argwhere(atlas_rois==0))

    if len(ROIsmask) <= 0:
        ROIsmask = atlas_rois

    x,y,z = atlas.get_fdata().shape
    
    image_final = np.squeeze(np.zeros((x,y,z,briks)))

    for r, roi in enumerate(ROIsmask):
        x_inds, y_inds, z_inds = np.where(atlas.get_fdata()==roi)
        
        image_final[x_inds, y_inds, z_inds] = results[:, r]

    img = image.new_img_like(atlas, image_final, affine=atlas.affine, copy_header=False)
    img.to_filename(savepath)

    return

if __name__ == '__main__': 
    
    # Set options and parameters
    condition = 'AV'    
    plot_clust = False
    plot_tsne = False
    metric = 'euclidean'
    plot_funcconn = False
    save_tsne = False
  
    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
    atlas_file = 'Schaefer200'
    rois_sign = np.loadtxt('{}cca_results/{}/group/fullmodel/significantROIs_{}.txt'.format(global_path, condition, condition)).astype(int)

    # Load functional connectivity matrix
    func_conn_distance_mat = 1 - np.loadtxt('{}/tSNE_clustering/func_conn/func_conn_matrix_{}signROIs.txt'.format(global_path, condition))

    # tSNE
    tsne = TSNE(random_state=0, metric='precomputed', init='random', perplexity=5, n_iter=5000)
    results_tsne = tsne.fit_transform(func_conn_distance_mat)

    # Save tSNE results
    if save_tsne:
        path = '{}tSNE_clustering/func_conn/'.format(global_path)
        if not os.path.exists(path):
            os.makedirs(path)

        np.savetxt('{}func_conn_tSNE_{}signROIs.txt'.format(path, condition), results_tsne)

    # Clustering
    tsne_distmat = squareform(pdist(results_tsne, metric=metric))

    n_clust_max = 10
    silhouette_avg, clusters_labels = evalKMeans(range(2, n_clust_max+1), tsne_distmat, print_otp=False)
    clust = np.argmax(silhouette_avg)

    cl_cols = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#001219', '#ffff99', '#b15928', '#990066', '#9ef01a', '#ffbc0a'] #['#696969', '#006400', '#4b0082', '#ff0000', '#00ced1', '#ffa500', '#ffff00', '#00ff00', '#00fa9a', '#0000ff', '#ff00ff', '#1e90ff', '#fa8072', '#eee8aa', '#ff1493']
    clust_colors = np.array([cl_cols[c-1] for c in clusters_labels[clust]])

    # Plot
    if plot_tsne:
        
        # Load ROI labels
        labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)[rois_sign-1]
        hemispheres = np.array([0 if l[0]=='R' else 1 for l in labels])
        labels_short = np.array([l[3:] for l in labels])
        right = np.where(hemispheres==0)[0]
        left = np.where(hemispheres)[0]

        # Create figure
        plt.figure(figsize=(9.6, 6.4))
        scatt = plt.scatter(results_tsne[right,0], results_tsne[right,1], c=clust_colors[right], marker='o')
        scattL = plt.scatter(results_tsne[left,0], results_tsne[left,1], c=clust_colors[left], marker='v')

        # Create legend
        legend_elements = [Line2D([0], [0], color='w', markerfacecolor=cl_cols[c], marker='o', label=c) for c in range(clust+2)]
        legend_elements.append(Line2D([0], [0], color='w', markerfacecolor='k', marker='o', label='right'))
        legend_elements.append(Line2D([0], [0], color='w', markerfacecolor='k', marker='v', label='left'))
                                    
        plt.legend(handles=legend_elements, loc='lower right', title='Clusters', ncol=2, frameon=False, borderpad=-1, labelspacing=0.1, borderaxespad=0.1, columnspacing=0.1, handletextpad=-0.5)
        plt.title('Clustering ROIs on tSNE space from functional connectivity distance matrix')
        plt.suptitle('Silhouette score for {} clusters = {}'.format(clust+2, silhouette_avg[clust]))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        for i, txt in enumerate(labels_short):
            ax.annotate(txt, (results_tsne[i,0]+0.5, results_tsne[i,1]+0.5), size='xx-small')

        path = '{}tSNE_clustering/func_conn/'.format(global_path)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig('{}tsne_{}_kmeans_{}_clusters.png'.format(path, metric, clust+2), dpi=300)

    # Save clustering res to nifti
    clust_to_nifti = np.expand_dims(clusters_labels[clust], axis=0)
    
    path = '{}tSNE_clustering/func_conn/'.format(global_path)
    if not os.path.exists(path):
        os.makedirs(path)
        
    save_nifti(clust_to_nifti, 1, atlas_file, '{}tsne_{}_kmeans_{}_clusters.nii.gz'.format(path, metric, clust+2), ROIsmask=rois_sign)