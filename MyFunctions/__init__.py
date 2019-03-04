#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 19:28:23 2018

@author: stefan
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from memory_profiler import profile
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from matplotlib import collections as mc
from numpy.random import uniform

#@profile
def Make_All_Cuts(DataFrame, 
                  cut_vars : "dict : {'Calo' : Calo_var_name, 'IsoCalo' : IsoCalo_var_name, ...}",
                  hard_cut_value : "Hard cut (sig) or [Hard cut (sig), Hard cut (bkg)]" = 0.5, 
                  soft_cut_value : "Soft cut (sig) or [Soft cut (sig), Soft cut (bkg)]" = 0.05,
                  pre_label : "MC / MC_trans / Data" = ""):
    
    base_sig = DataFrame.label0 >= 0.5
    base_bkg = DataFrame.label0 < 0.5
    
    df = pd.DataFrame(index=DataFrame.index.values)
    
    Calo_score     = cut_vars['Calo']
    ExtCalo_score  = cut_vars['ExtCalo']
    Iso_score      = cut_vars['Iso']
    IsoCalo_score  = cut_vars['IsoCalo']
    IsoTrack_score = cut_vars['IsoTrack']
    Track_score    = cut_vars['Track']
    ExtTrack_score = cut_vars['ExtTrack']
    
    if (type(hard_cut_value) is int) | (type(hard_cut_value) is float):
        hard_sig = hard_cut_value
        hard_bkg = 1 - hard_cut_value
    else:
        hard_sig = hard_cut_value[0]
        hard_bkg = hard_cut_value[1]
    
    if (type(soft_cut_value) is int) | (type(soft_cut_value) is float):
        soft_sig = soft_cut_value
        soft_bkg = 1 - soft_cut_value
    else:
        soft_sig = soft_cut_value[0]
        soft_bkg = soft_cut_value[1]
    
    hard_s = f"{int(100*hard_sig):02d}"
    soft_s = f"{int(100*soft_sig):02d}"
    
        # Cut only on label0
    colname = f'cut_label0{int(100*0.5):02d}'
    df[colname] = -1
    df.loc[base_sig, colname] = 1
    df.loc[base_bkg, colname] = 0
    
        # Cut on label0, Calo and Iso
    colname = f'{pre_label}_cut_Calo{hard_s}_Iso{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Calo_score] >= hard_sig) & 
                   (DataFrame[Iso_score]  >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Calo_score] < hard_bkg) &
                   (DataFrame[Iso_score]  < soft_bkg)), 
                  colname] = 0
                   
        # Cut on label0, Calo and IsoCalo
    colname = f'{pre_label}_cut_Calo{hard_s}_IsoCalo{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Calo_score]    >= hard_sig) & 
                   (DataFrame[IsoCalo_score] >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Calo_score]    < hard_bkg) &
                   (DataFrame[IsoCalo_score] < soft_bkg)), 
                  colname] = 0

        # Cut on label0, ExtCalo and Iso
    colname = f'{pre_label}_cut_ExtCalo{hard_s}_Iso{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtCalo_score] >= hard_sig) & 
                   (DataFrame[Iso_score]     >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtCalo_score] < hard_bkg) &
                   (DataFrame[Iso_score]     < soft_bkg)), 
                  colname] = 0
                   
        # Cut on label0, ExtCalo and IsoCalo
    colname = f'{pre_label}_cut_ExtCalo{hard_s}_IsoCalo{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtCalo_score] >= hard_sig) & 
                   (DataFrame[IsoCalo_score] >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtCalo_score] < hard_bkg) &
                   (DataFrame[IsoCalo_score] < soft_bkg)), 
                  colname] = 0
                  
        # Cut on label0, Calo and Track
    colname = f'{pre_label}_cut_Calo{hard_s}_Track{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Calo_score]  >= hard_sig) &
                   (DataFrame[Track_score] >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Calo_score]  < hard_bkg) & 
                   (DataFrame[Track_score] < soft_bkg)),
                  colname] = 0
    
        # Cut on label0, ExtCalo and ExtTrack
    colname = f'{pre_label}_cut_ExtCalo{hard_s}_ExtTrack{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtCalo_score]  >= hard_sig) &
                   (DataFrame[ExtTrack_score] >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtCalo_score]  < hard_bkg) & 
                   (DataFrame[ExtTrack_score] < soft_bkg)),
                  colname] = 0
        
        # Cut on label0, Track and Iso
    colname = f'{pre_label}_cut_Track{hard_s}_Iso{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Track_score] >= hard_sig) &
                   (DataFrame[Iso_score]   >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Track_score] < hard_bkg) & 
                   (DataFrame[Iso_score]   < soft_bkg)),
                  colname] = 0
                   
        # Cut on label0, Track and IsoTrack
    colname = f'{pre_label}_cut_Track{hard_s}_IsoTrack{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Track_score]    >= hard_sig) &
                   (DataFrame[IsoTrack_score] >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Track_score]    < hard_bkg) & 
                   (DataFrame[IsoTrack_score] < soft_bkg)),
                  colname] = 0
    
    # Cut on label0, ExtTrack and Iso
    colname = f'{pre_label}_cut_ExtTrack{hard_s}_Iso{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtTrack_score] >= hard_sig) &
                   (DataFrame[Iso_score]      >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtTrack_score] < hard_bkg) & 
                   (DataFrame[Iso_score]      < soft_bkg)),
                  colname] = 0
                   
        # Cut on label0, ExtTrack and IsoTrack
    colname = f'{pre_label}_cut_ExtTrack{hard_s}_IsoTrack{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtTrack_score] >= hard_sig) &
                   (DataFrame[IsoTrack_score] >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtTrack_score] < hard_bkg) & 
                   (DataFrame[IsoTrack_score] < soft_bkg)),
                  colname] = 0

        # Cut on label0, Track and Calo
    colname = f'{pre_label}_cut_Track{hard_s}_Calo{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Track_score] >= hard_sig) &
                   (DataFrame[Calo_score]  >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Track_score] < hard_bkg) & 
                   (DataFrame[Calo_score]  < soft_bkg)),
                  colname] = 0
                 
        # Cut on label0, ExtTrack and ExtCalo
    colname = f'{pre_label}_cut_ExtTrack{hard_s}_ExtCalo{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtTrack_score] >= hard_sig) &
                   (DataFrame[ExtCalo_score]  >= soft_sig)), 
                  colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtTrack_score] < hard_bkg) & 
                   (DataFrame[ExtCalo_score]  < soft_bkg)),
                  colname] = 0
        
        # Cut SOFT on label0, Calo, Iso and Track
    colname = f'{pre_label}_cut_Calo{soft_s}_Iso{soft_s}_Track{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Calo_score]  >= soft_sig) &
                   (DataFrame[Iso_score]   >= soft_sig) &
                   (DataFrame[Track_score] >= soft_sig)),
                   colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Calo_score]  < soft_bkg) &
                   (DataFrame[Iso_score]   < soft_bkg) &
                   (DataFrame[Track_score] < soft_bkg)),
                   colname] = 0
    
        # Cut HARD on label0, Calo, Iso and Track
    colname = f'{pre_label}_cut_Calo{hard_s}_Iso{hard_s}_Track{hard_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[Calo_score]  >= hard_sig) &
                   (DataFrame[Iso_score]   >= hard_sig) &
                   (DataFrame[Track_score] >= hard_sig)),
                   colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[Calo_score]  < hard_bkg) &
                   (DataFrame[Iso_score]   < hard_bkg) &
                   (DataFrame[Track_score] < hard_bkg)),
                   colname] = 0
    
        # Cut SOFT on label0, ExtCalo, Iso and ExtTrack
    colname = f'{pre_label}_cut_ExtCalo{soft_s}_Iso{soft_s}_ExtTrack{soft_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtCalo_score]  >= soft_sig) &
                   (DataFrame[Iso_score]      >= soft_sig) &
                   (DataFrame[ExtTrack_score] >= soft_sig)),
                   colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtCalo_score]  < soft_bkg) &
                   (DataFrame[Iso_score]      < soft_bkg) &
                   (DataFrame[ExtTrack_score] < soft_bkg)),
                   colname] = 0
    
        # Cut HARD on label0, ExtCalo, Iso and ExtTrack
    colname = f'{pre_label}_cut_ExtCalo{hard_s}_Iso{hard_s}_ExtTrack{hard_s}'
    df[colname] = -1
    df.loc[(base_sig & 
                   (DataFrame[ExtCalo_score]  >= hard_sig) &
                   (DataFrame[Iso_score]      >= hard_sig) &
                   (DataFrame[ExtTrack_score] >= hard_sig)),
                   colname] = 1
    df.loc[(base_bkg & 
                   (DataFrame[ExtCalo_score]  < hard_bkg) &
                   (DataFrame[Iso_score]      < hard_bkg) &
                   (DataFrame[ExtTrack_score] < hard_bkg)),
                   colname] = 0
    
    return df


#@profile
def combine_roc(y_true, score_a, score_b, sample_weight=None, 
                mask : "Used in data to choose signal and background, and not noneType" = None, 
                bins=1000, Range=(0,1), show_progress=True):
#    """ This function takes two scores and makes ROC curves for them both 
#        individually. It then combines the two ROC curves by looking at each
#        point in tpr for score_a, and calculating the tpr of score_b required
#        to reach the specified alpha."""
    sample_weight = 1 if sample_weight is None else sample_weight
    mask = np.ones(len(y_true)).astype(bool) if mask is None else mask
    
    a_fpr, a_tpr,_ = roc_curve(y_true[mask], score_a[mask],
                               sample_weight=sample_weight[mask],
                               drop_intermediate=False)
    b_fpr, b_tpr,_ = roc_curve(y_true[mask], score_b[mask],
                               sample_weight=sample_weight[mask],
                               drop_intermediate=False)
    
        # Find the lowest tpr, at which both roc curves have non-zero values
    lowest_common_tpr = max(min(a_tpr[a_tpr > 0]), min(b_tpr[b_tpr > 0]))
        # Delete fpr and tpr values for a and b that are lower than
        # the lowest common tpr. These values cannot be compared, 
        # and so are useless.
    a_fpr_cut = np.delete(a_fpr, np.argwhere(a_tpr < lowest_common_tpr))
    a_tpr_cut = np.delete(a_tpr, np.argwhere(a_tpr < lowest_common_tpr))
    b_fpr_cut = np.delete(b_fpr, np.argwhere(b_tpr < lowest_common_tpr))
    b_tpr_cut = np.delete(b_tpr, np.argwhere(b_tpr < lowest_common_tpr))
    
    if bins is not None:
        a_tpr_inter = np.linspace(max(lowest_common_tpr, Range[0]), min(Range[1], 1), bins)
        a_fpr_inter = np.interp(a_tpr_inter, a_tpr_cut, a_fpr_cut)
        b_tpr_inter = np.linspace(max(lowest_common_tpr, Range[0]), min(Range[1], 1), bins)
        b_fpr_inter = np.interp(b_tpr_inter, b_tpr_cut, b_fpr_cut)
        
        a_fpr_cut = a_fpr_inter
        a_tpr_cut = a_tpr_inter
        b_fpr_cut = b_fpr_inter
        b_tpr_cut = b_tpr_inter
    
    new_fpr = []
    new_tpr = []
    
        # Loop over signal efficiencies
    for i,sig_eff in enumerate(np.linspace(max(lowest_common_tpr, Range[0]), 
                               min(Range[1], 1), num=len(a_fpr_cut))):
        min_fpr = 1
        if show_progress:
            print(i)
        
            # Loop over tpr for score_a
        for j,_ in enumerate(a_tpr_cut):
            if a_tpr_cut[j] < sig_eff:
                continue
            else:
                index = np.argwhere(b_tpr_cut*a_tpr_cut[j] >= sig_eff)
                if len(index) == 0:
                    continue
                else:
                    index = index[0]
                        # Caluclate combined fpr
                    fpr_prod = (a_fpr_cut[j] * b_fpr_cut[index])[0]
                        # Set new min_fpr if it's lower than before
                    if fpr_prod < min_fpr:
                        min_fpr = fpr_prod
        
        new_fpr.append(min_fpr)
        new_tpr.append(sig_eff)
        
        
#    ab_tpr_cut = np.outer(a_tpr_cut, b_tpr_cut)
#    ab_fpr_cut = np.outer(a_fpr_cut, b_fpr_cut)
#    
#    for i,sig_eff in enumerate(np.linspace(lowest_common_tpr,1,num=len(a_fpr_cut))):
#            # Find indeces of first b_tpr where a_tpr*b_tpr >= sig_eff
#        indeces = np.sum(ab_tpr_cut < sig_eff, axis=0)
#        
#        
#        
#        new_fpr.append(min_fpr)
#        new_tpr.append(sig_eff)
            
    new_fpr = np.array(new_fpr)
    new_tpr = np.array(new_tpr)
    
    return new_fpr, new_tpr, a_fpr_cut, a_tpr_cut, b_fpr_cut, b_tpr_cut


def calc_total_weight(DataFrame, bkg_mask, et_reweighter, eta_reweighter, mu_reweighter):
    
    et_weight  = np.zeros(len(DataFrame))
    eta_weight = np.zeros(len(DataFrame))
    mu_weight  = np.zeros(len(DataFrame))
    
    et_weight[bkg_mask==1]  = et_reweighter.predict_weights(
                    DataFrame.loc[bkg_mask == 1, 'p_et_calo'])
    eta_weight[bkg_mask==1] = eta_reweighter.predict_weights(
                    DataFrame.loc[bkg_mask == 1, 'p_eta'])
    mu_weight[bkg_mask==1]  = mu_reweighter.predict_weights(
                    DataFrame.loc[bkg_mask == 1, 'averageInteractionsPerCrossing'])
    
    total_weight = et_weight*eta_weight*mu_weight
    
    return total_weight


def interpolation(values, X, Y, verbose=False):
    # values_trans = np.empty(shape=np.shape(values))
    # idx = np.empty(shape=(len(values),1))
    # for i,val in enumerate(values):
    #     idx[i] = np.where(val >= X)[0][-1]
    #     values_trans[i] = ((X[idx+1] - val) / (X[idx+1] - X[idx])) * (Y[idx+1] - Y[idx])
    # # values_trans = ((X[idx+1] - val) / (X[idx+1] - X[idx])) * (Y[idx+1] - Y[idx])
    # return values_trans
    
    if isinstance(values, pd.Series):
        values = values.values
    if isinstance(X, pd.Series):
        X = X.values
    if type(values) is float or type(values) is int:
        values = np.array([values,])
    
    idx = np.argmin(np.abs(values.reshape((1,-1)) - X.reshape(-1,1)), axis=0)
    idx[idx >= len(X)-1] -= 1
    values_trans = Y[idx] + ((X[idx+1] - values) / (X[idx+1] - X[idx])) * (Y[idx+1] - Y[idx])
    if verbose:
        print(f"idx: {max(idx)}")
        print(f"Values: {values}")
        print(f"Transformed values: {values_trans}\n")
    return values_trans
