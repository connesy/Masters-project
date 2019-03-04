#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:42:52 2018

@author: stefan
"""
import os
from time import time
import numpy as np
import pandas as pd
import h5py
import joblib
from hep_ml.reweight import BinsReweighter
from memory_profiler import profile

from contextlib import suppress

file_size = 1

def Build_Folder_Structure(directory_list):
    """ Create folders and parent folders recursively if they doesn't exist already."""
    for dirs in directory_list:
        with suppress(OSError):
            os.makedirs(dirs)
        
        # Old way:
        #try:
        #    os.makedirs(dirs)
        #except OSError:
        #    continue


def Get_Var_Lists():
        # Calorimeter variables
    Calo_vars  = ['p_Rhad1', 'p_Rhad', 'p_f3', 'p_weta2', 'p_Rphi', 'p_Reta', 
                 'p_Eratio', 'p_f1']
    ExtCalo_vars  = Calo_vars + ['p_eta', 'averageInteractionsPerCrossing']
        # Tracking variables
    Track_vars = ['p_numberOfInnermostPixelHits', 'p_numberOfPixelHits', 
                  'p_numberOfSCTHits', 'p_d0', 'p_d0Sig', 'p_dPOverP', 
                  'p_deltaEta1', 'p_deltaPhiRescaled2', 'p_EptRatio', 'p_TRTPID']
    ExtTrack_vars = Track_vars + ['p_numberOfTRTHits', 'p_numberOfTRTXenonHits']
        # Isolation variables
    # Iso_vars = ['p_etcone20', 'p_etcone30', 'p_etcone40',
    #         'p_etcone20ptCorrection', 'p_etcone30ptCorrection', 
    #         'p_etcone40ptCorrection', 'p_ptPU30',
    #         'p_ptcone20', 'p_ptcone30', 'p_ptcone40']
    IsoCalo_vars = ['p_etcone20', 'p_etcone30', 'p_etcone40',
                     'p_etcone20ptCorrection', 'p_etcone30ptCorrection', 
                     'p_etcone40ptCorrection', 'p_ptPU30']
    IsoTrack_vars = ['p_ptcone20', 'p_ptcone30', 'p_ptcone40']
    Iso_vars = IsoCalo_vars + IsoTrack_vars
        # Calorimiter+Tracking variables
    Calo_Track_vars = Calo_vars+Track_vars
    ExtCalo_Track_vars = ExtCalo_vars+ExtTrack_vars
    
    return [Calo_vars, ExtCalo_vars, Iso_vars, IsoCalo_vars, IsoTrack_vars,
            Track_vars, ExtTrack_vars, Calo_Track_vars, ExtCalo_Track_vars]

###############################################################################
# HFD5 storage            
###############################################################################
#@profile
def Load_DataFrame_from_hdf(Filename, compression=True):
    """ Load pandas DataFrame from .h5 storage."""
    Directory = os.path.abspath(os.path.dirname(Filename))
    file_ext = ".gzh5" if compression else ".h5"
    
        # Load dataframe from hdf files
    print("Loading DataFrame from hdf files...\n")
    t = time()
    DataFrame = pd.DataFrame()
    for f in sorted(os.listdir(Directory)):
        if f.find(os.path.basename(Filename)+"_") != -1 and f.find(file_ext) != -1:
            with h5py.File(Directory + "/" + f, 'r') as h5:
                df_temp = pd.DataFrame(h5[os.path.basename(Filename)][:])
                
            DataFrame = DataFrame.append(df_temp)
    
    assert(len(DataFrame) > 0)
    
    DataFrame.reset_index(drop=True, inplace=True)
    
    print(f"Loading DataFrame from hdf took {time()-t:.1f} seconds.\n")
    
    return DataFrame

#@profile
def Save_hdf(DataFrame, Filename, Num_of_files, compression):
    """ Saves a pandas DataFrame as Filename in a #Num_of_files series of .h5
        files for increased loading speed."""
    
    file_ext = ".gzh5" if compression else ".h5"
    
    df_numel = len(DataFrame.index)
    elems_per_file = int(np.floor(df_numel/Num_of_files))
    
    for i in range(Num_of_files):
        hdf_filename = Filename + f"_{i:02d}{file_ext}"
        with suppress(OSError):
            os.remove(hdf_filename)
        # try: os.remove(hdf_filename)
        # except: pass
        
        with h5py.File(hdf_filename) as h5:
            if i == Num_of_files - 1:
                h5.create_dataset(os.path.basename(Filename), 
                                      data=DataFrame.iloc[elems_per_file*i:, :].to_records(index=False), 
                                      compression=('gzip' if compression else None))
            else:
                h5.create_dataset(os.path.basename(Filename), 
                                  data=DataFrame.iloc[elems_per_file*i:elems_per_file*(i+1), :].to_records(index=False), 
                                  compression=('gzip' if compression else None))

#@profile
def hep_Reweight(DataFrame, Labels, save_weights=None, load_weights=None, is_MC=False):
    """ This function takes in a pandas DataFrame and reweighs E_t, eta and <mu>,
        by comparing the distributions in Bkg and Sig."""
        # Check that either save_weights or load_weights is None
    assert not (save_weights is not None and load_weights is not None), (
                    "Either save_weights or load_weights have to be None")
    if is_MC:
        et_label  = 'et_weight'
        eta_label = 'eta_weight'
        mu_label  = 'mu_weight'
        
        if load_weights is not None:
                # Load trained weight estimators
            et_reweighter  = joblib.load(load_weights+f"{et_label}.weight")
            eta_reweighter = joblib.load(load_weights+f"{eta_label}.weight")
            mu_reweighter  = joblib.load(load_weights+f"{mu_label}.weight")
            
        else:
                # Create weight estimators and fit them to DataFrame
            et_reweighter  = BinsReweighter(n_bins=200, n_neighs=3.)
            eta_reweighter = BinsReweighter(n_bins=200, n_neighs=1.)
            mu_reweighter  = BinsReweighter(n_bins=200, n_neighs=2.)
            
            et_reweighter.fit(original=DataFrame.loc[Labels < 0.5, 'p_et_calo'],
                       target=DataFrame.loc[Labels >= 0.5, 'p_et_calo'])
            eta_reweighter.fit(original=DataFrame.loc[Labels < 0.5, 'p_eta'],
                       target=DataFrame.loc[Labels >= 0.5, 'p_eta'])
            mu_reweighter.fit(original=DataFrame.loc[Labels < 0.5, 'averageInteractionsPerCrossing'],
                       target=DataFrame.loc[Labels >= 0.5, 'averageInteractionsPerCrossing'])
            
        
        et_weight = et_reweighter.predict_weights(DataFrame.loc[Labels < 0.5, 'p_et_calo'])
        et_weight = et_weight / np.mean(et_weight)
        DataFrame.loc[:, et_label] = 1
        DataFrame.loc[Labels < 0.5, et_label] = et_weight
        
        eta_weight = eta_reweighter.predict_weights(DataFrame.loc[Labels < 0.5, 'p_eta'])
        eta_weight = eta_weight / np.mean(eta_weight)
        DataFrame.loc[:, eta_label] = 1
        DataFrame.loc[Labels < 0.5, eta_label] = eta_weight
        
        mu_weight = mu_reweighter.predict_weights(DataFrame.loc[Labels < 0.5, 'averageInteractionsPerCrossing'])
        mu_weight = mu_weight / np.mean(mu_weight)
        DataFrame.loc[:, mu_label] = 1
        DataFrame.loc[Labels < 0.5, mu_label] = mu_weight
        
        ratio = np.sum(DataFrame.label0.values >= 0.5) / np.sum(DataFrame.label0.values < 0.5)
        
        total_weight = ratio * et_weight*eta_weight*mu_weight
        total_label = 'total_weight'
        DataFrame.loc[:, total_label] = 1
        DataFrame.loc[Labels < 0.5, total_label] = total_weight
        
        if save_weights is not None:
                    # Save BinsReweighter estimators fitted to training set
                joblib.dump(et_reweighter,  save_weights+f"{et_label}.weight")
                joblib.dump(eta_reweighter, save_weights+f"{eta_label}.weight")
                joblib.dump(mu_reweighter,  save_weights+f"{mu_label}.weight")
        
    elif not is_MC:
            # This section reweighs according to MC cuts
        for label in Labels:
                # MC cut label starts with "cut"
            if label.find("MC_cut") == 0:
                col_label = "MC_"
            elif label.find("MC_trans_cut") == 0:
                col_label = "MC_trans_"
                # Data cut labels start with "Data_cut"
            elif label.find("Data_cut") == 0:
                col_label = "Data_"
            else:
                continue

            ExtCalo_index = label.find("ExtCalo")
            Calo_index = label.find("Calo")
            if ExtCalo_index != -1:
                col_label += label[ExtCalo_index:ExtCalo_index+len("ExtCalo")+2] + "_"
            elif Calo_index != -1:
                col_label += label[Calo_index:Calo_index+len("Calo")+2] + "_"
            
            Iso_index = label.find("Iso")
            if Iso_index != -1:
                col_label += label[Iso_index:Iso_index+len("Iso")+2] + "_"
            
            ExtTrack_index = label.find("ExtTrack")
            Track_index = label.find("Track")
            if ExtTrack_index != -1:
                col_label += label[ExtTrack_index:ExtTrack_index+len("ExtTrack")+2] + "_"
            elif Track_index != -1:
                col_label += label[Track_index:Track_index+len("Track")+2] + "_"
                
            print(col_label)
            
            et_label    = col_label+"et_weight"
            eta_label   = col_label+"eta_weight"
            mu_label    = col_label+"mu_weight"
            total_label = col_label+"total_weight"
            
            if load_weights is not None:
                    # Load trained weight estimators
                et_reweighter  = joblib.load(load_weights+et_label)
                eta_reweighter = joblib.load(load_weights+eta_label)
                mu_reweighter  = joblib.load(load_weights+mu_label)
                
            else:
                    # Create weight estimators and fit them to DataFrame
                et_reweighter  = BinsReweighter(n_bins=200, n_neighs=3.)
                eta_reweighter = BinsReweighter(n_bins=200, n_neighs=1.)
                mu_reweighter  = BinsReweighter(n_bins=200, n_neighs=2.)
                
                et_reweighter.fit(
                        original=DataFrame.loc[np.abs(DataFrame[label].values) < 0.5, 'p_et_calo'],
                        target=DataFrame.loc[DataFrame[label] >= 0.5, 'p_et_calo'])
                eta_reweighter.fit(
                        original=DataFrame.loc[np.abs(DataFrame[label].values) < 0.5, 'p_eta'],
                        target=DataFrame.loc[DataFrame[label] >= 0.5, 'p_eta'])
                mu_reweighter.fit(
                        original=DataFrame.loc[np.abs(DataFrame[label].values) < 0.5, 'averageInteractionsPerCrossing'],
                        target=DataFrame.loc[DataFrame[label] >= 0.5, 'averageInteractionsPerCrossing'])
                
            #     # Et weights
            # et_weight = et_reweighter.predict_weights(
            #         DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'p_et_calo'])
            # DataFrame.loc[:, et_label] = -1
            # DataFrame.loc[DataFrame[label] >= 0.5, et_label] = 1
            # DataFrame.loc[np.abs(DataFrame[label]) < 0.5, et_label] = et_weight
            # DataFrame[et_label] = DataFrame[et_label].astype(np.float32)
    
            #     # Eta weights
            # eta_weight = eta_reweighter.predict_weights(
            #         DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'p_eta'])
            # DataFrame.loc[:, eta_label] = -1
            # DataFrame.loc[DataFrame[label] >= 0.5, eta_label] = 1
            # DataFrame.loc[np.abs(DataFrame[label]) < 0.5, eta_label] = eta_weight
            # DataFrame[eta_label] = DataFrame[eta_label].astype(np.float32)            

            #     # Mu weights
            # mu_weight = mu_reweighter.predict_weights(
            #         DataFrame.loc[np.abs(DataFrame[label]) < 0.5, 'averageInteractionsPerCrossing'])
            # DataFrame.loc[:, mu_label] = -1
            # DataFrame.loc[DataFrame[label] >= 0.5, mu_label] = 1
            # DataFrame.loc[np.abs(DataFrame[label]) < 0.5, mu_label] = mu_weight
            # DataFrame[mu_label] = DataFrame[mu_label].astype(np.float32)    

            #     # Total weights
            # total_weight = et_weight*eta_weight*mu_weight
            # DataFrame.loc[:, total_label] = -1
            # DataFrame.loc[DataFrame[label] >= 0.5, total_label] = 1
            # DataFrame.loc[np.abs(DataFrame[label]) < 0.5, total_label] = total_weight
            # DataFrame[total_label] = DataFrame[total_label].astype(np.float32)
            
                # Et weights
            et_weight = et_reweighter.predict_weights(
                    DataFrame.loc[DataFrame.label0 < 0.5, 'p_et_calo'])
            et_weight = et_weight / np.mean(et_weight)
            DataFrame.loc[:, et_label] = 1
            DataFrame.loc[DataFrame.label0 < 0.5, et_label] = et_weight
    
                # Eta weights
            eta_weight = eta_reweighter.predict_weights(
                    DataFrame.loc[DataFrame.label0 < 0.5, 'p_eta'])
            eta_weight = eta_weight / np.mean(eta_weight)
            DataFrame.loc[:, eta_label] = 1
            DataFrame.loc[DataFrame.label0 < 0.5, eta_label] = eta_weight

                # Mu weights
            mu_weight = mu_reweighter.predict_weights(
                    DataFrame.loc[DataFrame.label0 < 0.5, 'averageInteractionsPerCrossing'])
            mu_weight = mu_weight / np.mean(mu_weight)
            DataFrame.loc[:, mu_label] = 1
            DataFrame.loc[DataFrame.label0 < 0.5, mu_label] = mu_weight
            
            ratio = np.sum(DataFrame.label0.values >= 0.5) / np.sum(DataFrame.label0.values < 0.5)
            
                # Total weights
            total_weight = ratio * et_weight*eta_weight*mu_weight
            DataFrame.loc[:, total_label] = 1
            DataFrame.loc[DataFrame.label0 < 0.5, total_label] = total_weight
            
            if save_weights is not None:
                    # Save BinsReweighter estimators fitted to training set
                joblib.dump(et_reweighter,  save_weights+f"{et_label}.weight")
                joblib.dump(eta_reweighter, save_weights+f"{eta_label}.weight")
                joblib.dump(mu_reweighter,  save_weights+f"{mu_label}.weight")
    
    return DataFrame
    

def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print(f"Memory usage of properties dataframe is :{start_mem_usg} MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=False)  

            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes
            else:
                # if mn > np.finfo(np.float16).min and mx < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                elif mn > np.finfo(np.float64).min and mx < np.finfo(np.float64).max:
                    df[col] = df[col].astype(np.float64)   

            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print(f"Memory usage is: {mem_usg} MB")
    print(f"This is {mem_usg/start_mem_usg:.4%} of the initial size")
    return df, NAlist


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
