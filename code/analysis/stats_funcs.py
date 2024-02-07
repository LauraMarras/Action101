import numpy as np
from scipy.stats import pareto
from matplotlib import pyplot as plt


def pareto_right(distro, critical_p, tail_percentile=0.9, plot=False):

    null_distro = distro[1:]
    critical_value = distro[0]  

    permutations = null_distro.shape[0]
    q = np.percentile(null_distro, tail_percentile*100)

    right_tail = null_distro[null_distro>q] - q
    effect_eps = np.max(right_tail)/(permutations/len(right_tail))

    kHat, loc, sigmaHat = pareto.fit(right_tail, floc=0)

    if kHat > -0.5 and (critical_value-q) > 0:
        print('Estimating Pareto')
        
        estimated_cum = pareto.cdf(critical_value-q, kHat, loc, sigmaHat)
        pvalue = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))

        if pvalue < (np.finfo(np.float64).eps/2):
            bins = np.arange(0, critical_value-q, effect_eps)
            pvalue_emp = np.full((len(bins)), np.nan)

            for b, bin in enumerate(bins):
                estimated_cum = pareto.cdf(bin, kHat, loc, sigmaHat)
                pvalue_emp[b] = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))

            pvalue = (pvalue_emp[pvalue_emp > 0])[-1]

        if plot:
            plt.figure()
            
            plt.hist(right_tail, density=True)
            x = np.linspace(np.min(right_tail), np.max(right_tail)*1.5, 100)
            pdf = pareto.pdf(x, kHat, loc, sigmaHat)
            plt.plot(x, pdf)
           # plt.xlim(0, np.max(x))

            plt.axvline(critical_value-q)
            plt.show()

    else:
        print('Impossible to estimate pareto')
        distro_sort = np.sort(distro)
        # Get position of true result in sorted matrix for each domain
        positions = (np.where(distro_sort == critical_value))[0][-1]
        # Calculate pval based on position
        pvalue = 1-((positions)/(permutations+1))

    
    bins = np.arange(0, 5*np.max(right_tail), effect_eps/10)
    pvalue_emp = np.full((len(bins)), np.nan)
    for b, bin in enumerate(bins):
        estimated_cum = pareto.cdf(bin, kHat, loc, sigmaHat)
        pvalue_emp[b] = 1 - (tail_percentile + (estimated_cum*(1 - tail_percentile)))
    
    idx = (np.where(pvalue_emp > critical_p))[0][-1]
    critical_value_at_p = bins[idx]+q

    return pvalue, critical_value_at_p

def pvals(results, critical_p=0.05, pareto=False, plot=False):

    """
    Get p value for each domain
    
    Inputs:
    - results : array, 2d matrix of shape = n_perms by n_domains; containing R2 values for each permutation and each domain
    - pareto : wether to do pareto; default = False
    Outputs:
    - pvalues : array of shape = n_domains
    """

    # Define number of domains and permutations
    n_domains = results.shape[1]
    n_perms = results.shape[0]-1

    # Get true R
    r_true = results[0,:]

    if pareto:
        # Pareto
        pvals = np.full((n_domains), np.nan)
        critical_values_at_p = np.full((n_domains), np.nan)
        for d in range(n_domains):
            pvals[d], critical_values_at_p[d] = pareto_right(results[:,d], critical_p=critical_p, plot=plot)

    else:
        # Sort permuted results
        res_sorted = np.sort(results, axis=0)

        # Get position of true result in sorted matrix for each domain
        positions = np.array([np.where(np.isclose(res_sorted[:,d], r_true[d]))[0][0] for d in range(n_domains)])

        # Calculate pval based on position
        pvals = 1-((positions)/(n_perms+1))

        # Get critical values at p
        critical_values_at_p = np.percentile(results, (1-critical_p)*100, axis=0)

    return pvals, critical_values_at_p
