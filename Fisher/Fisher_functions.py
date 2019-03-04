#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:58:46 2018

@author: stefan
"""
import numpy as np
from time import time

def Make_Fisher_Coef(type_a, type_b, label):
    """ Calculate the Fisher coefficients for the two variable types 
        type_a and type_b."""
    
    sig_mask = label >= 0.5
    bkg_mask = np.abs(label) < 0.5
    
    sig_cov = np.cov(type_a[sig_mask], type_b[sig_mask])
    bkg_cov = np.cov(type_a[bkg_mask], type_b[bkg_mask])
    
    sig_mean = np.array([np.mean(type_a[sig_mask]), np.mean(type_b[sig_mask])])
    bkg_mean = np.array([np.mean(type_a[bkg_mask]), np.mean(type_b[bkg_mask])])
    
    fisher_w = np.dot(np.linalg.inv((sig_cov + bkg_cov)), sig_mean - bkg_mean)
    
    return fisher_w


def Fisher_Score(type_a, type_b, fisher_w):
    """ Calculate the Fisher 'score' for the combination of type_a and type_b.
        Score is moved and scaled to have range in [0;1]."""
    
    Fisher = np.column_stack((type_a, type_b)).dot(fisher_w.reshape(-1,1))
    Fisher -= Fisher.min()
    Fisher /= Fisher.max()
    
    return Fisher