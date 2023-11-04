import numpy as np
from scipy import special
from scipy.stats import kstwobign


def ks_weighted(arr1_all, arr2_all, w1_all, w2_all, return_dist=False):

    '''
    Given 2 arrays and their weights, returns Kolmogorov-Smirnov statistic and significance.

    This differs from the usual K-S test in that it computes a weighted K-S statistic and
    assumes the size of each sample is equal to the sum of the weights, not the length of
    the array. It's not exactly standard statistical practice to do this, so use with 
    caution, but it doesn't seem like a completely ridiculous idea, either.

        Parameters:
            arr1_all   (array): a data sample with values to be weighted by w1_all
            arr2_all   (array): a data sample with values to be weighted by w2_all
            w1_all     (array): weights for arr1_all
            w2_all     (array): weights for arr2_all
            return_dist (bool): True if the array of all K-S distances should be returned,
                                default is False (mostly only useful for debugging)

            Note: the arrays should be np.array() but other data types based on that should
                  work too, e.g. pd.Series, Astropy Table columns, etc. -- but if you get
                  an error on those, wrap your inputs with np.array().

        Returns:
            ks:     the weighted, 2-sided K-S statistic
            p_ks:   the p-value based on the weighted K-S statistic
            sig_ks: the significance level (in sigma) assuming p-values are distributed Normally

            if return_dist == True, also:
                dist_arr: array of KS-distances in raw format (sorted by increasing data sample value)
                          seriously this is not useful statistically, it's just for debugging

    '''

    # drop dead weight
    arr1 = np.array(arr1_all[w1_all > 0.0])
    arr2 = np.array(arr2_all[w2_all > 0.0])
    w1   = np.array(  w1_all[w1_all > 0.0])
    w2   = np.array(  w2_all[w2_all > 0.0])

    # get effective lengths of the weighted arrays
    n1 = np.sum(w1)
    n2 = np.sum(w2)
 
    # this is used below in the k-s calculation
    # (weighted sample sizes)
    ct = np.sqrt((n1+n2)/(n1*n2))

    # we want to sort the arrays, and the weights
    i1 = arr1.argsort()
    i2 = arr2.argsort()

    # sort arrays and weights in increasing order
    arr1_s = np.array(arr1[i1])
    w1_s   = np.array(  w1[i1])
    arr2_s = np.array(arr2[i2])
    w2_s   = np.array(  w2[i2])

    # make combined arrays but track which element comes from what, then sort them again
    both   = np.concatenate([arr1_s, arr2_s])
    both_w = np.concatenate([  w1_s,   w2_s])
    track  = np.concatenate([np.zeros(len(arr1_s), dtype=int), np.ones(len(arr2_s), dtype=int)])

    i_both   = both.argsort()
    both_s   = np.array(  both[i_both])
    both_w_s = np.array(both_w[i_both])
    track_s  = np.array( track[i_both])

    # go through array, once, computing the distance as we go, and track the max distance between cumulative curves
    # (which are both stored in the same array)
    # both cumulative curves start at 0 so the distance starts at 0
    # also cumulative curves always increase
    the_dist = 0.0
    dist_arr = np.zeros_like(both_s)
    max_dist = 0.0
    for j, this_which in enumerate(track_s):
        # the key here is the distance between curves goes up if array A has a new value,
        # and then if B has a new value that curve increases too so the curves get closer together
        # (the distance goes down).
        # it doesn't matter which is curve A and which is curve B, just that one increments 
        # and the other decrements.
        # if we were doing a regular K-S without weights, each new value for a given array changes
        # the distance between curves by 1 count. 
        # (with weighted, it only changes the distance by that object's weight.)
        # And also, these are cumulative curves, so each curve is divided by the total counts in that array
        # (which in the weighted case means the sum of the weights)
        # as a check, the distances should start at 0 and end at 0 (because the cumulative fractional
        # histograms both start at 0.0 and end at 1.0)
        if this_which == 0:
            the_dist += both_w_s[j]/n1
        else:
            the_dist -= both_w_s[j]/n2

        dist_arr[j] = the_dist
        if np.abs(the_dist) > max_dist:
            max_dist = np.abs(the_dist) 

    # the max dist over the whole cumulative curves is the K-S distance
    ks = max_dist
    # p-value (which also cares about the sample sizes)
    p_ks   = special.kolmogorov(float(ks)/float(ct))
    # scipy.stats.ks_2samp uses this instead?
    p_ksalt   = kstwobign.sf(((1./ct) + 0.12 + (0.11 * ct)) * ks)
    #print(p_ksalt)

    # what's the significance assuming a normal distribution? (1 = 1 sigma, 2. = 2 sigma, 3. = 3 sigma result etc.)
    sig_ks = special.erfcinv(p_ks)*np.sqrt(2.)



    if return_dist:
        return ks, p_ks, sig_ks, dist_arr
    else: 
        return ks, p_ks, sig_ks




# I don't really trust this one as you can get a different result with ks_w(x, y) and ks_w(y, x) which is NOT right
# Though note even the built-in scipy one seems to potentially have this problem because of how it sorts and how it computes distances
# so... I'm basically making an executive decision that it shouldn't matter and using the above
def ks_weighted_old(arr1_all, arr2_all, w1_all, w2_all, return_dist=False):

    # use with e.g.
    # print("K-S %.2e, p-value %.2e, i.e. %.1f sigma" % (ks, p_ks, sig_ks))

    # drop dead weight
    arr1 = arr1_all[w1_all > 0.0]
    arr2 = arr2_all[w2_all > 0.0]
    w1   =   w1_all[w1_all > 0.0]
    w2   =   w2_all[w2_all > 0.0]
    
    n1 = np.sum(w1)
    n2 = np.sum(w2)
    # this is used below in the k-s calculation
    # (weighted sample sizes)
    ct = np.sqrt((n1+n2)/(n1*n2))

    '''
    i1 = arr1.argsort()
    i2 = arr2.argsort()

    # sort arrays and weights in increasing order
    arr1_s = np.array(arr1[i1])
    w1_s   = np.array(  w1[i1])
    arr2_s = np.array(arr2[i2])
    w2_s   = np.array(  w2[i2])
    '''
    
    '''
    # from https://stackoverflow.com/a/40059727
    # uh, except it doesn't work? Dunno, but it's giving the wrong answers
    # I think this is because the answer on that page assumes, effectively, that you're already binned?
    # not sure, life's too short, my way takes 11 ms so wtf stop worrying about it
    data = np.concatenate([arr1_s, arr2_s])
    cwei1 = np.hstack([0, np.cumsum(w1_s)/sum(w1_s)])
    cwei2 = np.hstack([0, np.cumsum(w2_s)/sum(w2_s)])
    cdf1we = cwei1[[np.searchsorted(arr1, data, side='right')]]
    cdf2we = cwei2[[np.searchsorted(arr2, data, side='right')]]
    '''

    data = np.concatenate([arr1, arr2])

    
    #bins = np.linspace(np.min(data), np.max(data), 10*len(data))
    # if your data has ridiculous outliers this might need to be refined
    n_bins = 10*len(data)
    # histograms
    # this is where this is going wrong. You shouldn't have to bin anything!
    h1 = np.histogram(arr1, weights=w1, bins=n_bins)
    h2 = np.histogram(arr2, weights=w2, bins=n_bins)
    # cumulative + normalized
    cdf1we = np.hstack([0.0, np.cumsum(h1[0])/sum(h1[0])])
    cdf2we = np.hstack([0.0, np.cumsum(h2[0])/sum(h2[0])])

    # K-S distance
    ks = np.max(np.abs(cdf1we - cdf2we))
    # p-value
    p_ks   = special.kolmogorov(ks/ct)
    # scipy.stats.ks_2samp uses this instead?
    p_ksalt   = kstwobign.sf(((1./ct) + 0.12 + (0.11 * ct)) * ks)
    print(p_ksalt)
    # what's the significance assuming a normal distribution? (1 = 1 sigma, 2. = 2 sigma, 3. = 3 sigma result etc.)
    sig_ks = special.erfcinv(p_ks)*np.sqrt(2.)

    if return_dist:
        return ks, p_ks, sig_ks, np.abs(cdf1we - cdf2we)
    else: 
        return ks, p_ks, sig_ks




