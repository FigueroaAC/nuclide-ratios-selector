# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 12:02:26 2023

@author: afigue01
"""

from numpy import exp, array, zeros, linspace, sqrt, product, random, power
import matplotlib.pyplot as plt
from numba import njit, prange, float64, int32, types
from time import perf_counter
# In[]

def gaussian_pdf(x, mu, sd):
    return exp(-(x.T-mu)**2 / (2*(sd**2))).T

@njit(types.UniTuple(float64[:,:],2)(float64[:,:], 
                                     float64[:], 
                                     float64[:,:], 
                                     float64), 
      cache=True, parallel=True)
def calculate_marginals(data, likelihood, bins, tot_lkl):
    prob = zeros((data.shape[0], bins.shape[1]))
    for i in prange(data.shape[0]):
        for j in prange(bins.shape[1]):
            prob[i,j] = tot_lkl - likelihood[data[i,:]>=bins[i,j]].sum() \
                - prob[i,:].sum()
    
    return bins, prob

def summary_statistics(data, likelihood, tot_lkl):
    means = data.dot(likelihood/tot_lkl)
    sds = sqrt(power(data, 2).dot(likelihood/tot_lkl) - means**2)
    return means, sds

def analyze(data, likelihood, nbins, scale):
    tot_lkl = likelihood.sum()
    bins = linspace(data.min(axis=1), data.max(axis=1), nbins).T
    bins, probs = calculate_marginals(data=data, 
                                      likelihood=likelihood, 
                                      bins=bins,
                                      tot_lkl=tot_lkl)
    means,sds = summary_statistics(data=data, 
                                   likelihood=likelihood, 
                                   tot_lkl=tot_lkl)
    bins, means = scale * bins.T, scale * means.T
    return bins.T, probs, means.T, sds.T


# In[]

n_vars = 5
mn = 0
mx = 50

scale = random.uniform(low=mn, high=mx, size=n_vars)
x = random.uniform(size=n_vars)
s = random.uniform(low=0, high=0.1, size=n_vars)

actual_x, actual_s = scale * x, s

n = 10000000

xtest = random.uniform(size=(n_vars, n))

# TODO: rewrite this and alter everything so we work on sums of the
# loglikelihood and not products

lkl = gaussian_pdf(xtest, x, s).prod(axis=0)
nbins = 100
st = perf_counter()
bins, probs, means, sds = analyze(data=xtest,
                                  likelihood=lkl, 
                                  nbins=nbins,
                                  scale=scale)
end = perf_counter()
print(f'{end-st}')
for i in range(means.shape[0]):
    plt.figure()
    plt.plot(bins[i], probs[i], label='Marginal')
    plt.plot([means[i], means[i]], [0, probs[i].max()], label='True Value')
    print(actual_x[i], actual_s[i], means[i], sds[i])