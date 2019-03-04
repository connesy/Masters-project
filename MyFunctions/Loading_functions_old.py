#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:42:52 2018

@author: stefan
"""
import os
import time
import numpy as np
import pandas as pd
import h5py
from hep_ml.reweight import BinsReweighter

file_size = 1  # 0.8 Gibibytes

def Build_Folder_Structure(directory_list):
    """ Create folders and parent folders recursively if they doesn't exist already."""
    for dirs in directory_list:
        try:
            os.makedirs(dirs)
        except OSError:
            continue
    
    
def Get_Var_Lists():
        # Calorimeter variables
    Calo_vars  = ['p_Rhad1', 'p_Rhad', 'p_f3', 'p_weta2', 'p_Rphi', 'p_Reta', 
                 'p_Eratio', 'p_f1']
    Ext_Calo_vars  = Calo_vars + ['p_eta', 'averageInteractionsPerCrossing']
        # Tracking variables
    Track_vars = ['p_numberOfInnermostPixelHits', 'p_numberOfPixelHits', 
                  'p_numberOfSCTHits', 'p_d0', 'p_d0Sig', 'p_dPOverP', 
                  'p_deltaEta1', 'p_deltaPhiRescaled2', 'p_EptRatio', 'p_TRTPID']
    Ext_Track_vars = Track_vars + ['p_numberOfTRTHits', 'p_numberOfTRTXenonHits']
        # Isolation variables
    Iso_vars = ['p_etcone20', 'p_etcone30', 'p_etcone40',
            'p_etcone20ptCorrection', 'p_etcone30ptCorrection', 
            'p_etcone40ptCorrection', 'p_ptPU30',
            'p_ptcone20', 'p_ptcone30', 'p_ptcone40']
    Iso_Calo_vars = ['p_etcone20', 'p_etcone30', 'p_etcone40',
                     'p_etcone20ptCorrection', 'p_etcone30ptCorrection', 
                     'p_etcone40ptCorrection', 'p_ptPU30']
    Iso_Track_vars = ['p_ptcone20', 'p_ptcone30', 'p_ptcone40']
        # Calorimiter+Tracking variables
    Calo_Track_vars = Calo_vars+Track_vars
    
    return [Calo_vars, Ext_Calo_vars, Iso_vars, Iso_Calo_vars, Iso_Track_vars,
            Track_vars, Ext_Track_vars, Calo_Track_vars]

###############################################################################
# HFD5 storage            
###############################################################################

def Load_DataFrame_from_hdf(Filename, is_MC, compression=True, seed=None):
    """ Load pandas DataFrame from .h5 storage.
    
        If the .h5 file storage does not exist (e.g. first time the program is 
        run), the data is loaded from the corresponding CSV file. It is then
        converted to .h5 by CSV_to_hdf(), which reweighs and shuffles the 
        data before saving it."""

    Directory = os.path.abspath(os.path.dirname(Filename))
    hdf_filename = Directory + "/hdf_Files/" + os.path.basename(Filename)
    
    file_ext = ".gzh5" if compression else ".h5"
    
        # If hdf files do not exist, create them from csv file
    if not os.path.isfile(hdf_filename+"_00"+file_ext):
        CSV_to_hdf(Filename, is_MC, compression, seed)
    
        # Load dataframe from msp files
    print("Loading DataFrame from hdf files...\n")
    DataFrame = pd.DataFrame()
    for f in sorted(os.listdir(Directory + "/hdf_Files/")):
        if f.find(os.path.basename(Filename)+"_") != -1 and f.find(file_ext) != -1:
            with h5py.File(Directory + "/hdf_Files/" + f, 'r') as h5:
                df_temp = pd.DataFrame(h5[os.path.basename(Filename)][:])
                
            DataFrame = DataFrame.append(df_temp)
    
    DataFrame.reset_index(drop=True, inplace=True)
    
    print("Finished loading DataFrame from hdf files.\n")
    
    return DataFrame


def CSV_to_hdf(Filename, is_MC, compression, seed):
    """ Loads a CSV file from Filename, reweighs it in E_t, eta and <mu>, then
        shuffles it by the seed number and saves it as a series of .h5 files."""
    
    file_ext = ".gzh5" if compression else ".h5"
    
    if (Filename[-4:] == ".csv") | (Filename[-5:] == file_ext):
        Filename = Filename[:-4]

    CSV_filename_original = Filename + "_original.csv"
    CSV_filename = Filename + ".csv"
    
    Directory = os.path.abspath(os.path.dirname(Filename))
    hdf_filename = Directory + "/hdf_Files/" + os.path.basename(Filename)
    t_start = time.time()
    
    print("Converting " +os.path.basename(Filename)+" CSV to hdf...\n")
        # Read CSV into pandas DataFrame
    Loaded_DataFrame = pd.read_csv(CSV_filename_original)
    if is_MC:
            # Drop events that can neither be classified as signal nor background
        Loaded_DataFrame = TruthFix(Loaded_DataFrame, CSV_filename)
            # Reweigh DataFrame columns E_t, eta and <mu>
        Loaded_DataFrame = Reweight(Loaded_DataFrame, 
                                    Label=Loaded_DataFrame.Truth.values)
        Loaded_DataFrame = hep_Reweight(Loaded_DataFrame, 
                                        Label=Loaded_DataFrame.Truth.values)
    
        # Shuffle DataFrame by the provided seed to get rid of any ordering
    Loaded_DataFrame_shuffled = Loaded_DataFrame.sample(
            frac=1, random_state=seed)
        # Reset DataFrame index
    Loaded_DataFrame_shuffled.reset_index(drop=True, inplace=True)
    
    df_file_size = os.stat(CSV_filename_original).st_size/(1024**3)  # Size of CSV file in GB
    Num_of_files = int(df_file_size//file_size) + 1  # Number of files to break up .h5
    
         # Save DataFrame as a series of .h5 files to speed up loading times
    Save_hdf(Loaded_DataFrame_shuffled, hdf_filename, Num_of_files, compression)
    
    t_end = time.time()
    print("Converting files took",np.round(t_end - t_start, decimals=2),"seconds.\n")


def Save_hdf(DataFrame, Filename, Num_of_files, compression):
    """ Saves a pandas DataFrame as Filename in a #Num_of_files series of .h5
        files for increased loading times."""
    
    file_ext = ".gzh5" if compression else ".h5"
    
    df_numel = len(DataFrame.index)
    elems_per_file = int(np.floor(df_numel/Num_of_files))
    
    for i in range(Num_of_files):
        hdf_filename = Filename + "_{0:02d}".format(i) + file_ext
        if i == Num_of_files - 1:
            with h5py.File(hdf_filename) as h5:
                h5.create_dataset(os.path.basename(Filename), 
                                  data=DataFrame.iloc[elems_per_file*i:, :].to_records(index=False), 
                                  compression=('gzip' if compression else None))
        else:
            with h5py.File(hdf_filename) as h5:
                h5.create_dataset(os.path.basename(Filename), 
                                  data=DataFrame.iloc[elems_per_file*i:elems_per_file*(i+1), :].to_records(index=False), 
                                  compression=('gzip' if compression else None))
                

def TruthFix(DataFrame, CSV_filename):
    """ Removes events which can neither be classified as signal, nor
        as background."""
    mask = (((DataFrame.label0.values >= 0.5) & (DataFrame.p_TruthType == 2)) | 
            ((DataFrame.label0.values < 0.5) & (DataFrame.p_TruthType != 2)))
    
    new_DataFrame = DataFrame.loc[mask, :].copy(deep=True)
#    new_DataFrame = new_DataFrame.reset_index(drop=True)
    
    new_DataFrame.loc[((new_DataFrame.label0 >= 0.5) & 
                       (new_DataFrame.p_TruthType == 2)), 'Truth'] = 1
    new_DataFrame.loc[((new_DataFrame.label0 < 0.5) & 
                       (new_DataFrame.p_TruthType != 2)), 'Truth'] = 0
        
    new_DataFrame.to_csv(CSV_filename, index=False)
    
    return new_DataFrame


def Reweight(DataFrame, Label):
    """ This function takes in a pandas DataFrame and reweighs E_t, eta and <mu>,
        by comparing the distributions in Bkg and Sig."""
    
    et_bin_edges = np.concatenate((np.arange(15, 75 +1, 1)-0.5, 
                                   [100, 200, 300, 400, 500, 1000]))*1000  # Energy bin edges in MeV
    eta_bin_edges = np.arange(-2.47, 2.47 +1.5*0.02, 0.02)-0.01  # Eta bin edges
    mu_bin_edges = np.arange(12, 40 +1.5*2, 2)-1  # <mu> bin edges
    
    [et_sig_values, _] = np.histogram(DataFrame.loc[
            Label >= 0.5, 'p_et_calo'], 
            bins=et_bin_edges)
    [et_bkg_values, _] = np.histogram(DataFrame.loc[
            Label < 0.5, 'p_et_calo'], 
            bins=et_bin_edges)
            
    [eta_sig_values, _] = np.histogram(DataFrame.loc[
            Label >= 0.5, 'p_eta'], 
            bins=eta_bin_edges)
    [eta_bkg_values, _] = np.histogram(DataFrame.loc[
            Label < 0.5, 'p_eta'], 
            bins=eta_bin_edges)
            
    [mu_sig_values, _] = np.histogram(DataFrame.loc[
            Label >= 0.5, 'averageInteractionsPerCrossing'], 
            bins=mu_bin_edges)
    [mu_bkg_values, _] = np.histogram(DataFrame.loc[
            Label < 0.5, 'averageInteractionsPerCrossing'], 
            bins=mu_bin_edges)
    
        # Calculate the weight for each event and fix inf's
        # (inf is caused by division by 0, so no bkg events in the corresponding
        # bin. This means, that all events are sig, and so should have weight 1)
    et_weights = et_sig_values / et_bkg_values
    et_weights[np.isinf(et_weights)] = 1
    eta_weights = eta_sig_values / eta_bkg_values
    eta_weights[np.isinf(eta_weights)] = 1
    mu_weights = mu_sig_values / mu_bkg_values
    mu_weights[np.isinf(mu_weights)] = 1
    
    df = pd.DataFrame()
    
    et_index = np.sum(DataFrame.p_et_calo.values.reshape(1,-1) >= 
                      et_bin_edges[1:].reshape(-1, 1), axis=0)
    df['et_weight_old'] = et_weights[et_index]
    
    eta_index = np.sum(DataFrame.p_eta.values.reshape(1,-1) >= 
                       eta_bin_edges[1:].reshape(-1, 1), axis=0)
    df['eta_weight_old'] = eta_weights[eta_index]
    
    mu_index = np.sum(DataFrame.averageInteractionsPerCrossing.values.reshape(1,-1) >= 
                      mu_bin_edges[1:].reshape(-1, 1), axis=0)
    df['mu_weight_old'] = mu_weights[mu_index]
    
    df = df.set_index(DataFrame.index)

    sig_index = DataFrame[Label >= 0.5].index
    
    df.loc[sig_index, 'et_weight_old'] = 1
    df.loc[sig_index, 'eta_weight_old'] = 1
    df.loc[sig_index, 'mu_weight_old'] = 1
    
    df['total_weight_old'] = df.et_weight.values * df.eta_weight.values * df.mu_weight.values
    df.loc[sig_index, 'total_weight_old'] = 1
    
    DataFrame = pd.concat([DataFrame, df], axis=1)
        
    return DataFrame


def hep_Reweight(DataFrame, Labels, is_MC=True):
    """ This function takes in a pandas DataFrame and reweighs E_t, eta and <mu>,
        by comparing the distributions in Bkg and Sig."""
    if is_MC:
        et_reweighter = BinsReweighter(n_bins=200, n_neighs=3.)
        et_reweighter.fit(original=DataFrame.loc[Labels < 0.5, 'p_et_calo'],
                       target=DataFrame.loc[Labels >= 0.5, 'p_et_calo'])
        et_weight = et_reweighter.predict_weights(DataFrame.loc[Labels < 0.5, 'p_et_calo'])
        
        DataFrame.loc[:, 'et_weight'] = 1
        DataFrame.loc[Labels < 0.5, 'et_weight'] = et_weight
        
        eta_reweighter = BinsReweighter(n_bins=200, n_neighs=3.)
        eta_reweighter.fit(original=DataFrame.loc[Labels < 0.5, 'p_eta'],
                       target=DataFrame.loc[Labels >= 0.5, 'p_eta'])
        eta_weight = eta_reweighter.predict_weights(DataFrame.loc[Labels < 0.5, 'p_eta'])
        
        DataFrame.loc[:, 'eta_weight'] = 1
        DataFrame.loc[Labels < 0.5, 'eta_weight'] = eta_weight
        
        mu_reweighter = BinsReweighter(n_bins=200, n_neighs=3.)
        mu_reweighter.fit(original=DataFrame.loc[Labels < 0.5, 'averageInteractionsPerCrossing'],
                       target=DataFrame.loc[Labels >= 0.5, 'averageInteractionsPerCrossing'])
        mu_weight = mu_reweighter.predict_weights(DataFrame.loc[Labels < 0.5, 'averageInteractionsPerCrossing'])
        
        DataFrame.loc[:, 'mu_weight'] = 1
        DataFrame.loc[Labels < 0.5, 'mu_weight'] = mu_weight
        
        total_weight = et_weight*eta_weight*mu_weight
        
        DataFrame.loc[:, 'total_weight'] = 1
        DataFrame.loc[Labels < 0.5, 'total_weight'] = total_weight
    else:
        for label in Labels:
            if label.find("cut") != -1:
                col_label = ""
                if label.find("Ext_Calo") != -1:
                    col_label += "Ext_Calo_"
                elif label.find("Calo") != -1:
                    col_label += "Calo_"
                
                if label.find("Iso") != -1:
                    col_label += "Iso_"
                
                if label.find("Ext_Track") != -1:
                    col_label += "Ext_Track_"
                elif label.find("Track") != -1:
                    col_label += "Track_"
                
                print(col_label)
                
                et_label    = col_label+"et_weight"
                eta_label   = col_label+"eta_weight"
                mu_label    = col_label+"mu_weight"
                total_label = col_label+"total_weight"
                
                    # Et weights
                et_reweighter = BinsReweighter(n_bins=200, n_neighs=3.)
                et_reweighter.fit(original=DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'p_et_calo'],
                                  target=DataFrame.loc[DataFrame[label] >= 0.5, 'p_et_calo'])
                et_weight = et_reweighter.predict_weights(
                        DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'p_et_calo'])
                
                DataFrame.loc[:, et_label] = -1
                DataFrame.loc[DataFrame[label] >= 0.5, et_label] = 1
                DataFrame.loc[np.abs(DataFrame[label]) < 0.5, et_label] = et_weight
        
                    # Eta weights
                eta_reweighter = BinsReweighter(n_bins=200, n_neighs=3.)
                eta_reweighter.fit(original=DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'p_eta'],
                                  target=DataFrame.loc[DataFrame[label] >= 0.5, 'p_eta'])
                eta_weight = eta_reweighter.predict_weights(
                        DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'p_eta'])
                
                DataFrame.loc[:, eta_label] = -1
                DataFrame.loc[DataFrame[label] >= 0.5, eta_label] = 1
                DataFrame.loc[np.abs(DataFrame[label]) < 0.5, eta_label] = eta_weight
                
                    # Mu weights
                mu_reweighter = BinsReweighter(n_bins=200, n_neighs=3.)
                mu_reweighter.fit(original=DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'averageInteractionsPerCrossing'],
                                  target=DataFrame.loc[DataFrame[label] >= 0.5, 'averageInteractionsPerCrossing'])
                mu_weight = mu_reweighter.predict_weights(
                        DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'averageInteractionsPerCrossing'])
                
                DataFrame.loc[:, mu_label] = -1
                DataFrame.loc[DataFrame[label] >= 0.5, mu_label] = 1
                DataFrame.loc[np.abs(DataFrame[label]) < 0.5, mu_label] = mu_weight
        
                    # Total weights
                total_weight = et_weight*eta_weight*mu_weight
                
                DataFrame.loc[:, total_label] = -1
                DataFrame.loc[DataFrame[label] >= 0.5, total_label] = 1
                DataFrame.loc[np.abs(DataFrame[label]) < 0.5, total_label] = total_weight
    
    return DataFrame
    
    
###############################################################################
# Old msp storage
###############################################################################

#def Load_DataFrame_from_msp(Filename, is_MC=False, seed=None):
#    """ Load pandas DataFrame from .msp file series.
#    
#        If the .msp file series does not exist (e.g. first time the program is 
#        run), the data is loaded from the corresponding CSV file. It is then
#        converted to msp's by CSV_to_msp(), which reweighs and shuffles the 
#        data before saving them as msp's."""
#    
#    Directory = os.path.abspath(os.path.dirname(Filename))
#    msp_filename = Directory + "/msp_Files/" + os.path.basename(Filename)
#    
#        # If msp files do not exist, create them from csv file
#    if not os.path.isfile(msp_filename+"_00.msp"):
#        CSV_to_msp(Filename, is_MC, seed)
#    
#        # Load dataframe from msp files
#    print "Loading DataFrame from msp files...\n"
#    DataFrame = pd.DataFrame()
#    for f in sorted(os.listdir(Directory + "/msp_Files/")):
#                if f.find(os.path.basename(Filename)+"_") != -1:
#                    df_temp = pd.read_msgpack(Directory + "/msp_Files/" +f)
#                    DataFrame = DataFrame.append(df_temp)
#    print "Finished loading DataFrame from msp files.\n"
#    
#    return DataFrame
#
#
#def CSV_to_msp(Filename, is_MC, seed):
#    """ Loads a CSV file from Filename, reweighs it in E_t, eta and <mu>, then
#        shuffles it by the seed number and saves it as a series of .msp files."""
#    
#    if Filename[-4:] == ".csv":
#        Filename = Filename[:-4]
#
#    CSV_filename_original = Filename + "_original.csv"
#    CSV_filename = Filename + ".csv"
#    
#    Directory = os.path.abspath(os.path.dirname(Filename))
#    msp_filename = Directory + "/msp_Files/" + os.path.basename(Filename)
#    t_start = time.time()
#    
#    print "Converting " +Filename+" CSV to msp...\n"
#        # Read CSV into pandas DataFrame
#    Loaded_DataFrame = pd.read_csv(CSV_filename_original)
#        # Drop events that can neither be classified as signal nor background
#    Loaded_DataFrame_TruthFixed = TruthFix(Loaded_DataFrame, CSV_filename)
#        # Reweigh DataFrame columns E_t, eta and <mu>
#    if is_MC:
#        label = Loaded_DataFrame_TruthFixed.Truth.values
#    else:
#        label = Loaded_DataFrame_TruthFixed.label0.values
#        
#    Loaded_DataFrame_reweighted = Reweight(Loaded_DataFrame_TruthFixed, label)
#        # Shuffle DataFrame by the provided seed to get rid of any ordering
#    Loaded_DataFrame_reweighted_shuffled = Loaded_DataFrame_reweighted.sample(
#            frac=1, random_state=seed)
#    
#    df_file_size = os.stat(CSV_filename).st_size/(1024**3)  # Size of CSV file in GB
#    Num_of_files = int(df_file_size//file_size) + 1  # Number of files to break up .msp
#    
#         # Save DataFrame as a series of .msp files to speed up loading times
#    Save_msp(Loaded_DataFrame_reweighted_shuffled, msp_filename, Num_of_files)
#    
#    t_end = time.time()
#    print "Converting files took",np.round(t_end - t_start, decimals=2),"seconds.\n"
#    
#        
#def Save_msp(DataFrame, Filename, Num_of_files):
#    """ Saves a pandas DataFrame as Filename in a #Num_of_files series of .msp 
#        files for increased loading times."""
#    
#    df_numel = len(DataFrame.index)
#    elems_per_file = int(np.floor(df_numel/Num_of_files))
#    
#    for i in range(Num_of_files):
#        msp_filename = Filename + "_{0:02d}".format(i) + ".msp"
#        if i == Num_of_files - 1:
#            DataFrame.iloc[elems_per_file*i:, :].to_msgpack(msp_filename)
#        else:
#            DataFrame.iloc[elems_per_file*i:elems_per_file*(i+1), :].to_msgpack(msp_filename)