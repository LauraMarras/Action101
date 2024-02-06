import numpy as np
import sys

from simul_pipeline import simulate_subject

if __name__ == '__main__':
    print('Starting')
    orig_stdout = sys.stdout

    # Define options
    sub_list = [5] #[*range(0,5)]
    n_subs = len(sub_list)
    options = {'add_noise_bool': True, 'add_trend_bool': False, 'add_motion_bool': False, 'save': False}

    # Define fMRI parameters
    TR = 2
    np.random.seed(0)
    R = np.random.uniform(0.2, 0.71, n_subs)
    betas = None # In this case, we generate random betas # 1
    n_bins_betas = 15
    noise_level = 4
    n_tissues = 4 # air, white matter, grey matter, csf
    n_bins_trend = 80
    semination_mask = 'mask_2orig.nii.gz' # In this case, we seminate the same roi in all subjects (manually created in advance) # 'roi_109.nii' 

    # Define motion parameters
    movement_upscale = 1
    regressors_path = 'data/simulazione_datasets/motionreg/'
    
    # Define task parameters
    domain = 'agent_objective' # In this case we seminate the same model in all subjects
    task_path = 'data/models/domains/group_us_conv_{}.csv'.format(domain)
    task_time_res = 0.05
    run_dur_sec = [536,450,640,650,472,480] # duration of each run in seconds
    run_dur_TR = (np.array(run_dur_sec)/TR).astype(int) # duration of each run in TRs
    run_cuts = np.array(list(zip((np.cumsum(run_dur_TR) - run_dur_TR), np.cumsum(run_dur_TR))))
    n_runs = len(run_dur_sec)
    
    # Create seed schema
    seed_schema = np.reshape(np.arange(0,((n_runs*2)+2)*n_subs), (n_subs,n_runs*2+2))

    # Sub loop
    for s, sub in enumerate(sub_list):

        fmri_params = {'TR': TR, 'R': R[s], 'betas': betas, 'n_bins_betas': n_bins_betas, 'noise_level': noise_level, 'n_tissues': n_tissues, 'n_bins_trend': n_bins_trend, 'semination_mask': semination_mask} # In this case, the goal R is different for each subject
        task_params = {'task_path': task_path, 'task_time_res': task_time_res, 'n_runs': n_runs, 'run_cuts': run_cuts} # In this case, the task regressors are identical for each subject
        motion_params = {'movement_upscale': movement_upscale, 'regressors_path': regressors_path}

        # Print output to txt file
        logfile = open('data/simulazione_results/logs/out_sub-{}.txt'.format(sub+1), 'w')
        sys.stdout = logfile

        print('Simulation of sub-{}'.format(sub+1))
        print('- task parameters: {}'.format(task_params))
        print('- fMRI parameters: {}'.format(fmri_params))
        print('- seed schema: {}'.format(seed_schema[s]))
        print('- options: {}'.format(options))

        # Call Pipeline
        simulate_subject(sub+1, fmri_params, task_params, motion_params, seed_schema[s], options)
    
        # Close textfile
        logfile.close()

    # Close logfile
    sys.stdout = orig_stdout

    print('finished')