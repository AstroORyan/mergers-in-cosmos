import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, FormatStrFormatter
import seaborn as sns
import numpy as np
import scipy as sp
import numpy as np
from scipy import special
from scipy.stats import kstwobign

plt.rc('text', usetex = True)

##Sets the plot style
plt.style.use("SebaStyle")
font= {"family":"Times New Roman", "size": 12}
plt.rc("font", **font)

data=pd.read_csv('MergedDataSetsVolLimLessColumns.csv',sep=',')

def weightscalculations():
    massbins=np.arange(9.25,11.5,0.1)
    masscolumn='lmass_x'

    for i in range(len(massbins)-1):
        print(i)
        subset=(data[masscolumn]>massbins[i]) & (data[masscolumn]<=massbins[i+1])
        subset=data[subset]
        offsetbarnumber=subset['offsetbars'].sum()
        barnumber=subset['bars'].sum()-offsetbarnumber
        
        

        for column,row in data.iterrows(): #goes through all the row numbers (labeled as 'row') and all the column names 
            mass=row['lmass_x']
            offset=row['offsetbars']
            bar=row['bars']
            if offsetbarnumber>barnumber:
                weight=barnumber/offsetbarnumber
                if (mass>massbins[i]) & (mass<=massbins[i+1]) & (offset==True):
                    data.loc[column,'massweight']=weight
                else:
                    continue

            else:
                weight=offsetbarnumber/barnumber
                if (mass>massbins[i]) & (mass<=massbins[i+1]) & (offset==False) &(bar==True):
                    data.loc[column,'massweight']=weight
                else:
                    continue      
        
    data.to_csv('MergedDataSetsVolLimLessColumns.csv')

def histnotprob(property,xlabel,xmin,xmax,binwidth,limits):
    weight = 'massweight'
    alldiscs=data['discs']==True
    alldiscs=data[alldiscs]

    allbarred=(data['bars']==True) &(data['offsetbars']==False)
    allbarred=data[allbarred]

    alloffsetbarred=data['offsetbars']==True
    alloffsetbarred=data[alloffsetbarred]
    
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Normalized Fraction')
    if limits==True:
        ax1.set_xlim(xmin,xmax)

    sns.histplot(ax=ax1,data=alloffsetbarred,x=property,weights=weight,element='step',fill=False,stat='probability',binwidth=binwidth,label='Offset Bars',color='#a57ecc',ls='dashed') #
    sns.histplot(ax=ax1,data=allbarred,x=property,weights=weight,element='step',fill=False,stat='probability',binwidth=binwidth,label='Centered Bars',color='#e14b55')
    sns.histplot(ax=ax1,data=alldiscs,x=property,element='step',weights=weight,fill=True,alpha=0.4,stat='probability',binwidth=binwidth,label='Discs',color='#91c8fa',edgecolor='#82b4e1')

    #KS test 
    # print('bar v offset '+str(sp.stats.ks_2samp(alloffsetbarred[property],allbarred[property])))
    # print('bar v discs '+str(sp.stats.ks_2samp(alldiscs[property],allbarred[property])))
    # print('discs v offset '+str(sp.stats.ks_2samp(alloffsetbarred[property],alldiscs[property])))

    # #Anderson-Darling tests
    # print('bar v offset '+str(sp.stats.anderson_ksamp([alloffsetbarred[property],allbarred[property]])))
    # print('bar v discs '+str(sp.stats.anderson_ksamp([alldiscs[property],allbarred[property]])))
    # print('discs v offset '+str(sp.stats.anderson_ksamp([alloffsetbarred[property],alldiscs[property]])))

    ax1.legend(loc='upper right',fontsize='small')

    plt.show()
    
def weightedkstest(arr1_all,arr2_all,w1_all,w2_all,return_dist=False):
    # drop dead weight
    # arr1 = arr1_all[w1_all > 0.0]
    # arr2 = arr2_all[w2_all > 0.0]
    # w1   =   w1_all[w1_all > 0.0]
    # w2   =   w2_all[w2_all > 0.0]

    arr1=arr1_all
    arr2=arr2_all
    w1=w1_all
    w2=w2_all

    # get effective lengths of the weighted arrays
    n1 = np.sum(w1)
    n2 = np.sum(w2)
 
    # this is used below in the k-s calculation
    # (weighted sample sizes)
    ct = np.sqrt((n1+n2)/(n1*n2))

    # we want to sort the arrays, and the weights
    i1 = arr1.argsort()
    i2 = arr2.argsort()

    arr1_s=[]
    w1_s=[]
    arr2_s=[]
    w2_s=[]
    # sort arrays and weights in increasing order
    for i in i1:
        arr1_s.append(arr1[i])
        w1_s.append(w1[i])
    
    for i in i2:
        arr2_s.append(arr2[i])
        w2_s.append(w2[i])

    # make combined arrays but track which element comes from what, then sort them again
    both   = np.concatenate([arr1_s, arr2_s])
    both_w = np.concatenate([w1_s,w2_s])
    track  = np.concatenate([np.zeros(len(arr1_s), dtype=int), np.ones(len(arr2_s), dtype=int)])

    i_both   = both.argsort()
    both_s   = np.array(both[i_both])
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
        print(ks,p_ks,sig_ks)
        return ks, p_ks, sig_ks


def arraysforweightedKS(property):
    weight = 'massweight'

    alldiscs=data['discs']==True
    alldiscs=data[alldiscs]
    discweight=alldiscs[weight]

    allbarred=(data['bars']==True) &(data['offsetbars']==False)
    allbarred=data[allbarred]
    barweight=allbarred[weight]

    alloffsetbarred=data['offsetbars']==True
    alloffsetbarred=data[alloffsetbarred]
    offsetbarweight=alloffsetbarred[weight]

    discpropvalues=alldiscs[property]
    barpropvalues=allbarred[property]
    offsetbarpropvalues=alloffsetbarred[property]

    return discpropvalues,barpropvalues,offsetbarpropvalues,discweight,barweight,offsetbarweight


# weightscalculations()
histnotprob('lSFR_tot_x',r'weighted log(SFR) $[M_{\odot}yr^{-1}]$',9.25,11.5,0.2,False) # For Mass input: 'lmass_x',r'$log(M(M_*))[M_\odot]$' , for SFR: 'lSFR_tot_x',r'weighted log(SFR) $[M_{\odot}yr^{-1}]$'
discpropvalues,barpropvalues,offsetbarpropvalues,discweight,barweight,offsetbarweight=arraysforweightedKS('lSFR_tot_x')
print('barsVoffsets')
weightedkstest(barpropvalues,offsetbarpropvalues,barweight,offsetbarweight,return_dist=False)