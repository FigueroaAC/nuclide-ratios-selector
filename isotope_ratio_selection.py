#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:25:38 2019

@author: AFigueroa
"""

from scipy.interpolate import interp2d
from numpy import ndarray, array, exp, vstack, sqrt, divide, zeros, linspace, load, round as npround
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def get_Y_matrix(Y: ndarray, NB: int):
    SortedY = vstack(([Y[i*25 : (i*25) + 25] for i in range(NB)]))
    return SortedY     

def gaussian_pdf(x:array, target:float, sigma:float):
    return exp(-(x - target)**2 / (2 * (sigma**2)))

def get_isotopes(list_element: str):
    Ratios = list_element.split('-')
    return Ratios

def Uncertainty(Isotope_Ratio):
    """Statistical Uncertainty Estimation for a given isotope ratio of R, assumming that the amount of atoms of 
    the isotope with least concentration is 1e10"""
    return sqrt(((1/Isotope_Ratio)**2 + (1/Isotope_Ratio)) * (1/1e10))

def Posterior_Unc_Calculation(Isotope_Combinations,Y,N_of_B,N_of_Ct,size):
    #Output_Dictionary = {}
    for Isotope_Combination in tqdm(Isotope_Combinations):
        # Separate the Ratios
        Ratio1 = Isotope_Combination[0]
        Ratio2 = Isotope_Combination[1]
        # Obtain the Isotopes that make each Ratio
        IsoR1 = get_isotopes(Ratio1)
        IsoR2 = get_isotopes(Ratio2)
        # Load the Data Matrices
        M11 = get_Y_matrix(Y[IsoR1[0]], len(N_of_B))
        M12 = get_Y_matrix(Y[IsoR1[1]], len(N_of_B))
        M21 = get_Y_matrix(Y[IsoR2[0]], len(N_of_B))
        M22 = get_Y_matrix(Y[IsoR2[1]], len(N_of_B))
        print(M11.shape)
        # Calculate the ratio Matrices:
        R1 = zeros((M11.shape[0], M11.shape[1]))
        for i in range(shape(R1)[0]):
            for j in range(shape(R1)[1]):
                if M11[i][j] < M12[i][j]:
                    R1[i][j] = divide(M11[i][j], M12[i][j])
                else:
                    R1[i][j] = divide(M12[i][j], M11[i][j])
        R2 = zeros((M21.shape[0], M21.shape[1]))
        for i in range(R2.shape[0]):
            for j in range(R2.shape[1]):
                if M21[i][j] < M22[i][j]:
                    R2[i][j] = divide(M21[i][j], M22[i][j])
                else:
                    R2[i][j] = divide(M22[i][j], M21[i][j])
        # Create the interpolator functions:
        funcs = [interp2d(N_of_Ct, N_of_B, R1), interp2d(N_of_Ct, N_of_B, R2)]
        # Create the set of test points where regression is made to evaluate the uncertainty:
        Btest = linspace(0, 50, int(size))
        Ctest = linspace(0, 21600, int(size))
        # Create solubility matrix, we are interested in the regions of space for which the largest posterior
        # uncertainty is smaller or equal to the largest ratio-based measurement uncertainty
        Solubility_Matrix = zeros((size, size))
        # Calculate the expected posterior uncertainty based on Approximate Bayes Computation:
        Bsamples = linspace(0, 50, 1000)
        Ctsamples = linspace(0, 21600, 1000)
        for i in range(size):
            for j in range(size):
                # Define query points where the posterior uncertainty is calculated:
                Bsol = Btest[i]
                Ctsol = Ctest[j]
                # Calculate what are the expected values of the respective functions at these points:
                Ytarget = [funcs[0](Ctsol, Bsol), funcs[1](Ctsol, Bsol)]
                # Calculate the expected statistical measurement
                Unc  = [Uncertainty(Ytarget[a]) for a in range(len(Ytarget))]
                sigma = [Unc[a] * Ytarget[a] for a in range(len(Ytarget))]
                # Calculate the posterior probability:
                prob = gaussian_pdf(funcs[0](Ctsamples, Bsamples), Ytarget[0], sigma[0]) * gaussian_pdf(funcs[1](Ctsamples, Bsamples), Ytarget[1], sigma[1])
                # Compute and Normalize Marginals:
                MarginalB = prob.sum(axis=0) * (Ctsamples[1] - Ctsamples[0])
                MarginalB = MarginalB / MarginalB.sum()
                MarginalCt = prob.sum(axis=1) * (Bsamples[1] - Bsamples[0])
                MarginalCt = MarginalCt / MarginalCt.sum()
                # Calculate Statistics:
                Bmean = Bsamples.dot(MarginalB)
                Bstd = sqrt(((Bsamples - Bmean)**2).dot(MarginalB))
                Ctmean = Ctsamples.dot(MarginalCt)
                Ctstd = sqrt(((Ctsamples - Ctmean)**2).dot(MarginalCt))
                uncB = 100 * Bstd / Bmean
                uncCt = 100 * Ctstd / Ctmean
                maxUnc = max([uncB, uncCt])
                Solubility_Matrix[i][j] = maxUnc
        # Store a sparse form of the matrix in the form of only the indexes where it is solvable

        return Solubility_Matrix

print('Loading Training Data')
Xtrain = load('', allow_pickle=True)
Ytrain = load('', allow_pickle=True).item()
N_of_B = sorted(list(set([x[0] for x in Xtrain])))
N_of_Ct = sorted(list(set(npround([x[1] for x in Xtrain], 2))))
print('Training Data Loaded')
    
Isotope_Ratios = load('ratios.npy', allow_pickle=True)

Rtest = get_isotopes(Isotope_Ratios[0])
Ratio_Comb = [[Isotope_Ratios[i], Isotope_Ratios[j]] for i in range(len(Isotope_Ratios)) for j in range(i+1, len(Isotope_Ratios))]
size = 10
Output = Posterior_Unc_Calculation(Ratio_Comb[:1], Ytrain, N_of_B, N_of_Ct, size)
