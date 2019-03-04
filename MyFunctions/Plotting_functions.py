#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:51:39 2018

@author: stefan
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
palette = sns.color_palette()
from sklearn.metrics import roc_curve, auc#, roc_auc_score

def ATLAS(ax, MC='Both', test=True, x=0.05, y=0.95, 
          verticalalignment='top', horizontalalignment='left',
          fontsize=14):
    """ Add a box on the ax saying "ATLAS Work in progress"\n"MC/Data.
        Returns the created matplotlib.text.Text object."""
    props = dict(boxstyle='square', facecolor='white', edgecolor='white', alpha=0)
    if MC == 'both':
        textstr = r"$\bf{ATLAS}$ Work in progress" + "\nMC & Data" + f" {'test sets' if test else 'training sets'}"
    else:
        textstr = r"$\bf{ATLAS}$ Work in progress" + f"\n{'MC' if MC else 'Data'}" + f" {'test set' if test else 'training set'}"
    
    text = ax.text(x, y, textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=verticalalignment, horizontalalignment=horizontalalignment,
            bbox=props)
    
    return text


def Histogram(Data, sig_mask, bkg_mask, weights, bins=100, log=True, sig_color='r',
              bkg_color='b', legend=None, title='', xlabel='', ylabel='',
              save=None):
    """"""
    
    if legend is not None:
        legend_label = ", "+legend
    else:
        legend_label = ""
    
    plt.hist(Data[sig_mask], weights=weights[sig_mask], bins=bins, histtype='step', 
             color=sig_color, label='Signal'+legend_label)
    plt.hist(Data[bkg_mask], weights=weights[bkg_mask], bins=bins, histtype='step', 
             color=bkg_color, label='Background'+legend_label)
    
    if log: plt.yscale('log', nonposy='clip')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.show(block=0)
    
    if save is not None:
        plt.savefig(save+"Hist_"+xlabel.replace(" ","_")+f"_bins={bins}"+
                    ("_log")*log+".png", dpi=100)
    

def ROC(Data_scores, Truth, weights, bkg_type='acc', label='', title='',
                xlim=(0.8,1), ylim='auto', log=True, figname=None,
                color='k', linestyle='-', linewidth=2,
                grid=True, grid_linestyle='--', legend=True, legend_loc='best',
                alpha_marker=0.92, save=None, ax=None):
    """"""
    if grid: plt.rc(('grid', grid_linestyle))
    if figname is None:
        figname = label+"_log"*log
    
    fpr, tpr, _ = roc_curve(Truth, Data_scores, sample_weight=weights)
    
    if alpha_marker is not None:
        x_val = tpr[tpr >= alpha_marker][0]
        if bkg_type == 'acc':
            y_val = fpr[tpr >= alpha_marker][0]
        else:
            y_val = 1 - fpr[tpr >= alpha_marker][0]
        alpha_val = int(100*alpha_marker)
        label += ", " +"1-"*(bkg_type!="acc")+r"$\alpha_{" + f"{alpha_val:2d}" + r"}" + fr" = {100*y_val:.3f}\%$"
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(14,9), constrained_layout=True)
    
    if bkg_type == 'acc':
            # Plot background acceptance
        ax.plot(tpr, fpr, color=color, label=label, linestyle=linestyle,
                 linewidth=linewidth)
    else:
            # Plot background rejection
        ax.plot(tpr, 1-fpr, color=color, label=label, linestyle=linestyle,
                 linewidth=linewidth)
    
    ax.set_xticks(np.arange(xlim[0], xlim[1]+(xlim[1]-xlim[0])/10, (xlim[1]-xlim[0])/10))
    ax.set_xlim(xlim)
    
    if log:
            # Set plot y-axis to log10 scale
        ax.set_yscale('log', nonposy='clip')
            # Set y-axis ticks to one power of 10 lower than the lowest plotted value
        if bkg_type == 'acc':
            if ylim == 'auto':
                ylim = (pow(10,np.ceil(np.log10(min(fpr[tpr >= xlim[0]]))-1)), 1)
        else:
            if ylim == 'auto':
                ylim = (pow(10,-1),1)
    else:
        if ylim == 'auto':
            ax.set_yticks(np.arange(ylim[0], ylim[1]+(ylim[1]-ylim[0])/10, (ylim[1]-ylim[0])/10))
    
    ax.set_ylim(ylim)
    
    # if alpha_marker is not None:
    #     ax.vlines(x_val, ylim[0], y_val, colors=color,
    #                linestyle='dashed', label=None)
    #     ax.hlines(y_val, xlim[0], x_val, colors=color,
    #                linestyle='dashed', label=None)
        
        
    if grid: ax.grid(True)
    
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background " + (r'acceptance ($\alpha$)' if bkg_type == 'acc' else r'rejection ($1-\alpha$)'))
    ax.set_title(title)
    if legend:
        ax.legend(loc=legend_loc)
    plt.draw()
    plt.show(block=0)
    
    if save is not None:
        if alpha_marker is not None:
            plt.savefig(save+"ROC_"+figname.replace(" ","_")+"_"+bkg_type+
                        f"_x=({xlim[0]:.2f},{xlim[1]:.2f})_y=({ylim[0]:.2f},{ylim[1]:.2f})_alpha={alpha_val:2d}.png", dpi=100)
        else:
            plt.savefig(save+"ROC_"+figname.replace(" ","_")+"_"+bkg_type+
                        f"_x=({xlim[0]:.2f},{xlim[1]:.2f})_y=({ylim[0]:.2f},{ylim[1]:.2f}).png", dpi=100)
    
    roc_auc = auc(fpr, tpr, reorder=True)
    
    return roc_auc, (y_val if alpha_marker is not None else None)
    

def alpha_1D(Data_scores, Truth, weights, cut_var, cut_values, sig_eff, 
             bkg_type='acc', label='', xlabel='', ylabel='', 
             log=False, color='b', save=None):
    """ Plot background acceptance/rejectance @ 92% signal efficiency for 
        different cuts."""
    
    figname = label+"_log"*log
    
    bkg_acc_rej = []
    for i,_ in enumerate(cut_values):
        if i < len(cut_values)-1:
            mask = ((cut_var >= cut_values[i]) & 
                    (cut_var < cut_values[i+1]))
            
            weight = weights[mask] if weights is not None else None
            
            alpha_i = _calc_alpha(
                        Data_scores[mask],
                        Truth[mask],
                        weight,
                        sig_eff)        
            if bkg_type == 'acc':
                bkg_acc_rej.append(alpha_i)
            else:
                bkg_acc_rej.append(1 - alpha_i)
            
#    plt.bar(cut_values[:-1], height=bkg_acc_rej, width=1000,
#            align='edge', log=log, label=label, color=color,
#            facecolor='none', edgecolor=color)
    plt.step(cut_values[:-1], bkg_acc_rej, label=label, color=color)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel+", log"*log)
    if log:
        plt.yscale('log', nonposy='clip')
    plt.title("Background " + ('acc.' if bkg_type == 'acc' else 'rej.') +
               f" @ {sig_eff:.2f}% signal efficiency")
    plt.legend()
    plt.tight_layout()
    plt.draw()
    plt.show(block=0)
    
    if save is not None:
        plt.savefig(save+"1D_"+"1-"*(bkg_type!='acc')+
                    f"sig_eff:{int(sig_eff*100):2d}_"+
                    figname.replace(" ","_")+".png", dpi=100)
    
    return bkg_acc_rej

def _calc_alpha(scores, truth, weights, sig_eff = 0.92,
                drop_intermediate=False):
    """ Internal function used to calculate background acceptance (alpha) 
        @ sig_eff (signal efficiency). Used in alpha_1D and alpha_2D."""
    fpr, tpr, _ = roc_curve(truth, scores, sample_weight=weights, 
                            drop_intermediate=drop_intermediate)
    
    alpha = fpr[tpr >= sig_eff][0]
    
    return alpha





    