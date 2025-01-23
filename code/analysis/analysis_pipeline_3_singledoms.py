import os
import numpy as np
from nilearn import image

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
    conditions = ['AV', 'vid', 'aud']
    save = False
    save_nifti_opt = False
    atlas_file = 'Schaefer200'
    alpha = 0.05
    maxT = True

    global_path = '/home/laura.marras/Documents/Repositories/Action101/data/'
    domains_list = np.array(['space', 'effector', 'agent & object', 'social', 'emotion', 'linguistic'])

    for condition in conditions:

        # Filter ROIs with significant R2 for full model
        pvals_group_fm, results_group_fm, rois_list = np.load('{}cca_results/{}/group/fullmodel/CCA_R2_pvals{}_group_{}.npz'.format(global_path, condition, '_maxT' if maxT else '', atlas_file), allow_pickle=True).values()
        ROIs_sign_idx = np.unique(np.where(np.squeeze(pvals_group_fm) < alpha)[0])
        ROIs_sign = rois_list[ROIs_sign_idx]

        # Load atlas labels and coords and filter for only tested ROIs
        roi_labels = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer_7N200_labels.txt', dtype=str)[rois_list-1]
        roi_coords = np.loadtxt('/data1/Action_teresi/CCA/atlas/Schaefer200_CdM.txt').round().astype(int)[rois_list,:]*np.array([-1, -1, 1]) # Convert from RAI to LPI

        # Get descriptives
        r2_avg_allrois = np.mean(results_group_fm).round(4)
        r2_std_allrois = np.std(results_group_fm).round(4)
        r2_max = np.max(results_group_fm).round(4)
        r2_max_roilab = roi_labels[np.argmax(results_group_fm)]
        r2_max_roi = rois_list[np.argmax(results_group_fm)]
        r2_max_coords = roi_coords[np.argmax(results_group_fm)]
        r2_avg_signrois = np.mean(results_group_fm[ROIs_sign_idx]).round(4)
        r2_std_signrois = np.std(results_group_fm[ROIs_sign_idx]).round(4)

        results_text = '''
        Results full model {}:
        Average R2 across all tested rois (N={}) is {} ± {}
        Highest R2 is {}, in ROI_{} {} at x={}, y={}, z={}; LPI
        Average R2 across significant rois (N={}) is {} ± {}
        '''.format(condition,
                   len(rois_list), r2_avg_allrois, r2_std_allrois,
                   r2_max, r2_max_roi, r2_max_roilab, r2_max_coords[0], r2_max_coords[1], r2_max_coords[2],
                   len(ROIs_sign), r2_avg_signrois, r2_std_signrois)
        
        # Load R2 from full model and variance partitioning, select first perm (true R), and significative ROIs
        results_vp = np.load('{}cca_results/{}/group/variancepart/CCA_R2_allsubs_{}.npz'.format(global_path, condition, atlas_file), allow_pickle=True)['results'][:, ROIs_sign_idx, 0, :]
        results_fm = np.load('{}cca_results/{}/group/fullmodel/CCA_R2_pvals_allsubs_{}.npz'.format(global_path, condition, atlas_file), allow_pickle=True)['results'][:, ROIs_sign_idx, 0, :]
    
        # Get R2 for single domain (subtract shuffled models R2 from full model R2)
        res_dom = np.squeeze(results_fm - results_vp)

        # Get mean across subs and concatenate to main matrix
        res_dom_mean = np.mean(res_dom, axis=0)
        res_dom_final = np.concatenate((res_dom, np.expand_dims(res_dom_mean, axis=0)), axis=0)

        # Get descriptives
        r2_max_doms = np.max(res_dom_mean, axis=0).round(4)
        r2_max_doms_roi = rois_list[ROIs_sign_idx][np.argmax(res_dom_mean, axis=0)]
        r2_max_doms_roilab = roi_labels[ROIs_sign_idx][np.argmax(res_dom_mean, axis=0)]
        r2_max_doms_coords = roi_coords[ROIs_sign_idx][np.argmax(res_dom_mean, axis=0)]
        r2_avg_doms = np.mean(res_dom_mean, axis=0).round(4)
        r2_std_doms = np.std(res_dom_mean, axis=0).round(4)
        percent_fm = ((r2_avg_doms*100)/r2_avg_signrois).round(1)
        percent_shared = (((r2_avg_signrois-np.sum(r2_avg_doms))*100)/r2_avg_signrois).round(1)
        order = np.argsort(r2_avg_doms)[::-1]

        for d, dom in enumerate(domains_list[order]):
            results_text += '''
            Results domain {}:
            - Average R2 is {} ± {}
            - {}% of full model R²
            - Highest R2 is {}, in ROI_{} {} at x={}, y={}, z={}; LPI
            '''.format(dom, r2_avg_doms[order][d], r2_std_doms[order][d], percent_fm[order][d],
                        r2_max_doms[order][d], r2_max_doms_roi[order][d], r2_max_doms_roilab[order][d], r2_max_doms_coords[order][d,0], r2_max_doms_coords[order][d,1], r2_max_doms_coords[order][d,2])
            
        results_text += '''
        Explained variance shared across multiple domains: {}%
        '''.format(percent_shared)

        # Save results
        if save:
            
            path = '{}/cca_results/{}/group/single_doms/'.format(global_path, condition)
            if not os.path.exists(path):
                os.makedirs(path)

            with open('{}cca_results/{}/group/results_{}.txt'.format(global_path, condition, condition), 'w') as res_file:
                res_file.write(results_text)
                
            np.savetxt('{}cca_results/{}/group/fullmodel/significantROIs_{}.txt'.format(global_path, condition, condition), ROIs_sign)
            np.savez('{}CCA_R2_group_singledoms'.format(path), results_group_sd=res_dom_mean, rois_list=ROIs_sign)
            np.savez('{}CCA_R2_allsubs_singledoms'.format(path), results_subs_sd=res_dom, rois_list=ROIs_sign)
            
        # Save nifti
        if save_nifti_opt:
            for dom in range(res_dom.shape[-1]):
                save_nifti(res_dom_final[:,:,dom], res_dom_final.shape[0], atlas_file, '{}CCA_R2_{}_{}.nii'.format(path, domains_list[dom], atlas_file), ROIs_sign)
