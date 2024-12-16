import os
import numpy as np
from nilearn import image
import matplotlib.pyplot as plt
from canonical_correlation_funcs import extract_roi

def config_plt(textsize=8):

    plt.rc('font', family='Arial')
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=textsize)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)

    plt.rc('legend', fontsize=textsize)
    plt.rc('legend', loc='best')
    plt.rc('legend', frameon=False)

    plt.rc('grid', linewidth=0.5)
    plt.rc('axes', linewidth=0.5)
    plt.rc('xtick.major', width=0.5)
    plt.rc('ytick.major', width=0.5)


if __name__ == '__main__': 
    
    conditions = ['AV', 'aud', 'vid']
    sub_lists = {'AV': np.array([12, 13, 14, 15, 16, 17, 18, 19, 22, 32]), 'vid': np.array([20, 21, 23, 24, 25, 26, 28, 29, 30, 31]), 'aud': np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 27])}
    n_conds = len(conditions)
    n_subs = sub_lists[conditions[0]].shape[0]

    # Load atlas ROI labels
    atlas_file = 'Schaefer200'
    roislabels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)
    n_rois = len(roislabels)

    # Initialize matrix to store results for each condition
    exp_var_conds = np.full((n_subs, n_rois, n_conds), np.nan)
    n_voxels_conds = np.full((n_subs, n_rois, n_conds), np.nan)
    r2 = np.full((n_rois, n_conds), np.nan)

    # Set figure and axes
    config_plt()
    fig, axs = plt.subplots(n_conds, 1, dpi=300, figsize=(8.3, (2.5)*n_conds))

    # Load data for each condition
    for c, condition in enumerate(conditions):

        sub_list = sub_lists[condition]
        n_subs = len(sub_list)
        global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    
        rois_to_include = list(np.loadtxt('{}cca_results/AV/group/fullmodel/significantROIs_AV.txt'.format(global_path)).astype(int)) if condition != 'AV' else [*range(1,201)]
        n_rois = len(rois_to_include)

        # Load R2
        r2[np.array(rois_to_include)-1,c] = np.squeeze(np.load('{}cca_results/{}/group/fullmodel/CCA_R2_pvals_maxT_group_Schaefer200.npz'.format(global_path, condition))['results_group'])

        # Load data
        try:
            pca_res = np.load('{}pca/pca_expvar_allsubs_{}.npy'.format(global_path, condition))
            n_voxels_mat = np.load('{}pca/pca_nvox_allsubs_{}.npy'.format(global_path, condition))

        except FileNotFoundError:
            # Initialize matrices
            pca_res = np.full((n_subs, n_rois), np.nan)
            n_voxels_mat = np.full((n_subs, n_rois), np.nan)
        
            # Get explained variance and n_voxels for each subject
            for s, sub_ in enumerate(sub_list):
                
                sub = str(sub_) if len(str(sub_))>=2 else '0'+str(sub_)

                # Load data
                data = image.load_img('/data1/ISC_101_setti/dati_fMRI_TORINO/sub-0{}/ses-{}/func/allruns_cleaned_sm6_SG.nii.gz'.format(sub, condition)).get_fdata()
                atlas = image.load_img('/data1/Action_teresi/CCA/atlas/sub-{}_{}_atlas_2orig.nii.gz'.format(sub, atlas_file)).get_fdata()
                
                # Extract rois
                _, _, n_voxels = extract_roi(data, atlas, rois_to_include)
                n_voxels_mat[s] = np.array(list(n_voxels.values()))
                
                # Load R2 results of single subject
                res_sub = np.load('{}cca_results/{}/sub-{}/fullmodel/CCA_res_sub-{}_{}.npz'.format(global_path, condition, sub, sub, atlas_file), allow_pickle=True)['pca_dict'].item()
                pca_res[s] = np.array([val[-1] for key, val in res_sub.items()])
        
            # Save
            np.save('{}pca/pca_expvar_allsubs_{}.npy'.format(global_path, condition), pca_res)
            np.save('{}pca/{}.npy'.format(global_path, condition), n_voxels_mat)

        # Fill global matrix
        exp_var_conds[:, np.array(rois_to_include)-1, c] = pca_res
        n_voxels_conds[:, np.array(rois_to_include)-1, c] = n_voxels_mat

        # Get avg and CI across subs
        exp_var_avg = np.mean(pca_res, axis=0)
        exp_var_min = np.min(pca_res, axis=0)
        exp_var_max = np.max(pca_res, axis=0)

        n_vox_avg = np.mean(n_voxels_mat, axis=0)
        n_vox_min = np.min(n_voxels_mat, axis=0)
        n_vox_max = np.max(n_voxels_mat, axis=0)

        # Plot
        if condition == 'AV':
            n_vox_avg_order = np.copy(n_vox_avg)
            rois_to_include_order = np.copy(rois_to_include)
            idxs_order = np.argsort(n_vox_avg_order[np.array(rois_to_include)-1])[::-1]
        
        idxs = np.argsort(n_vox_avg_order[np.array(rois_to_include)-1])[::-1]
        
        # Plot explained variance
        ax = axs[c]
        ax.plot(rois_to_include, exp_var_avg[idxs], linewidth=0.3, marker='.', markersize=3, label='explained variance')
        ax.fill_between(rois_to_include, exp_var_min[idxs], exp_var_max[idxs], alpha=0.3)

        ax.set_ylabel('Explained variance')
        ax.set_ylim([0.65,1])
        ax.spines['top'].set_visible(False)
        ax.margins(x=0)

        # Add axis and plot n_voxels
        ax2 = ax.twinx()
        ax2.plot(rois_to_include, n_vox_avg[idxs], linewidth=0.3, marker='.', markersize=3, color='red', label='number of voxels')
        ax2.fill_between(rois_to_include, n_vox_min[idxs], n_vox_max[idxs], alpha=0.3, color='red', edgecolor=None)

        ax2.set_ylabel('Number of voxels')
        ax2.set_ylim([0, 500])
        ax2.spines['top'].set_visible(False)
        ax2.margins(x=0)
        
        # Plot horizontal line indicating min number of voxels
        ax2.axhline(38, color='red', ls='--', linewidth=0.3)

    # Add x axis ticks and labels
    axs[0].set_xticks([])
    axs[1].set_xticks([])
    ax.set_xlabel('ROIs')
    ax.set_xticks(ticks=rois_to_include_order, labels=roislabels[idxs_order], rotation=90, size=3)
    
    # Add legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[0].legend(lines + lines2, labels + labels2, loc='right', frameon=False)

    plt.tight_layout()
    plt.savefig('{}pca/plots/explainedvariance_conditions.png'.format(global_path))

    # Second Plot
    expvar = np.nanmean(exp_var_conds, axis=(0,2))
    nvox = np.nanmean(n_voxels_conds, axis=(0,2))
    expvar_max = np.nanmax(exp_var_conds, axis=(0,2))
    nvox_max = np.nanmax(n_voxels_conds, axis=(0,2))
    expvar_min = np.nanmin(exp_var_conds, axis=(0,2))
    nvox_min = np.nanmin(n_voxels_conds, axis=(0,2))

    r2_avg = np.nanmean(r2, axis=-1)
    r2_max = np.nanmax(r2, axis=1)
    r2_min = np.nanmin(r2, axis=1)

    fig, axs = plt.subplots(2, 1, dpi=300, figsize=(8.3, 5), sharex=True)

    # Plot explained variance
    x = np.arange(1, 201)
    ax = axs[0]
    idxs = np.argsort(nvox)[::-1]
    idxs2exclude = np.where(np.isnan(r2[idxs,-1]))[0]

    ax.plot(x, expvar[idxs], linewidth=0.3, marker='.', markersize=3, color='#1f77b4', label='explained variance')
    ax.plot(x[idxs2exclude], expvar[idxs][idxs2exclude], color='#8fbbda', linestyle='', marker=".", markersize=3)
    
    ax.fill_between(x, expvar_min[idxs], expvar_max[idxs], alpha=0.3, color='#8fbbda', edgecolor=None)

    ax.set_ylabel('Explained variance')
    ax.set_ylim([0.65,1])
    ax.spines['top'].set_visible(False)
    ax.margins(x=0)

    # Add axis and plot n_voxels
    ax2 = ax.twinx()
    ax2.plot(x, nvox[idxs], linewidth=0.3, marker='.', markersize=3, color='#720026', label='number of voxels')
    ax2.plot(x[idxs2exclude], nvox[idxs][idxs2exclude], color='#ce4257', linestyle='', marker=".", markersize=3)
    
    ax2.fill_between(x, nvox_min[idxs], nvox_max[idxs], alpha=0.3, color='#ce4257', edgecolor=None)

    ax2.set_ylabel('Number of voxels')
    ax2.set_ylim([0, 500])
    ax2.spines['top'].set_visible(False)
    ax2.margins(x=0)
    
    # Plot horizontal line indicating min number of voxels
    ax2.axhline(38, color='#4f000b', ls='--', linewidth=0.5)

    # Add x axis ticks and labels
    #ax.set_xlabel('ROIs')
    ax.set_xticks(ticks=[])#rois_to_include_order, labels=roislabels[idxs], rotation=90, size=3)
  
    # Add legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='right', frameon=False)

    # Plot R2
    ax= axs[1]
    
    ax.plot(x, r2_avg[idxs], linewidth=0.3, marker='.', markersize=3, color='#226f54', label='R2')
    ax.plot(x[idxs2exclude], r2_avg[idxs][idxs2exclude], color='#87c38f', linestyle='', marker=".", markersize=3)
    
    ax.fill_between(x, r2_min[idxs], r2_max[idxs], alpha=0.3, color='#87c38f', edgecolor=None)

    ax.set_ylabel('R2')
    ax.set_ylim([0, 0.1])
    ax.spines['top'].set_visible(False)
    ax.margins(x=0)

    # Add axis and plot n_voxels
    ax2 = ax.twinx()
    ax2.plot(x, nvox[idxs], linewidth=0.3, marker='.', markersize=3, color='#720026', label='number of voxels')
    ax2.plot(x[idxs2exclude], nvox[idxs][idxs2exclude], color='#ce4257', linestyle='', marker=".", markersize=3)
    
    ax2.fill_between(x, nvox_min[idxs], nvox_max[idxs], alpha=0.3, color='#ce4257', edgecolor=None)

    ax2.set_ylabel('Number of voxels')
    ax2.set_ylim([0, 500])
    ax2.spines['top'].set_visible(False)
    ax2.margins(x=0)
    
    # Plot horizontal line indicating min number of voxels
    ax2.axhline(38, color='#4f000b', ls='--', linewidth=0.5)

    # Add x axis ticks and labels
    ax.set_xlabel('ROIs')
    ax.set_xticks(ticks=[])#, labels=roislabels[idxs], rotation=90, size=3)
  
    # Add legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig('{}plots/explainedvariance_R2_mean.png'.format(global_path))
