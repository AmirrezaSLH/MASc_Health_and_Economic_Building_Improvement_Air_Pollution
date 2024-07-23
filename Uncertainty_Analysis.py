# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:47:46 2024

@author: asalehi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geopandas as gpd

import json

from shapely.geometry import Polygon


from matplotlib.path import Path


from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from matplotlib.patches import PathPatch

from matplotlib.colors import LinearSegmentedColormap

from statistics import mean, median

#%%

import main

#%%

def code_comply_all( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    if  (climatezone_grid_map[GRID_KEY] == 1) or (climatezone_grid_map[GRID_KEY] == 2) :
        if ach50_segment_dict['SF_V1'] > 5:
            ach50_segment_dict['SF_V1'] = 5
        if ach50_segment_dict['SF_V2'] > 5:
            ach50_segment_dict['SF_V2'] = 5
        if ach50_segment_dict['SF_V3'] > 5:
            ach50_segment_dict['SF_V3'] = 5
    else:
        if ach50_segment_dict['SF_V1'] > 3:    
            ach50_segment_dict['SF_V1'] = 3
        if ach50_segment_dict['SF_V2'] > 3:
            ach50_segment_dict['SF_V2'] = 3
        if ach50_segment_dict['SF_V3'] > 3:
            ach50_segment_dict['SF_V3'] = 3
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention

#%%


def aggregate_results( dCin, dM, benefits, dY, costs, net_benefits, n):
    pop_dict = main.init_population()
    dCin_national = []
    dM_national = []
    dY_national = []
    benefits_national = []
    costs_national = []
    net_benefits_national = []

    for i in range(n):
        dCin_national.append(  main.national_average(dCin[i], pop_dict) )
        dM_national.append( sum(dM[i].values()) )
        dY_national.append ( main.national_average(dY[i], pop_dict) * 100000)
        costs_national.append( sum( costs[i].values()) / 1000000000)
        benefits_national.append( sum( benefits[i].values()) / 1000000000)
        net_benefits_national.append( sum(net_benefits[i].values()) / 1000000000 )
        
    return dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national

def analysis_stats(data):
    avg = mean(data)
    minimum = min(data)
    maximum = max(data)
    p2_5 = np.percentile(data, 2.5)
    p97_5 = np.percentile(data, 97.5)
    
    print('mean: ', avg)
    print('min: ', minimum)
    print('max: ', maximum)
    print('p2.5: ', p2_5)
    print('p97.5: ', p97_5)
    
    return 0
#%%

n = 5000
rR = 0.0058

def scenario_UC1(re_risk, int_func, itr, hf = 'HF_3'):
    dCin, dM, benefits, dY, costs, net_benefits, finf0, finf1, achn0, achn1 = main.run_MCS(  intervention_function = int_func, rr = re_risk , iterations = itr, HF = hf)
    dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national = aggregate_results( dCin, dM, benefits, dY, costs, net_benefits, n)
    
    return dCin, dM, benefits, dY, costs, net_benefits, dCin_national, dM_national, benefits_national, dY_national, costs_national, net_benefits_national, finf0, finf1, achn0, achn1


s1_Cin, s1_dM, s1_benefits, s1_dY, s1_costs, s1_nb, s1_Cin_mean, s1_dM_mean, s1_benefits_mean, s1_dY_mean, s1_costs_mean, s1_nb_mean, s1_finf0, s1_finf1, s1_achn0, s1_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')

s2_Cin, s2_dM, s2_benefits, s2_dY, s2_costs, s2_nb, s2_Cin_mean, s2_dM_mean, s2_benefits_mean, s2_dY_mean, s2_costs_mean, s2_nb_mean, s2_finf0, s2_finf1, s2_achn0, s2_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s2_dM_mean)
analysis_stats(s2_benefits_mean)
plt.boxplot(s2_dM_mean)

s3_Cin, s3_dM, s3_benefits, s3_dY, s3_costs, s3_nb, s3_Cin_mean, s3_dM_mean, s3_benefits_mean, s3_dY_mean, s3_costs_mean, s3_nb_mean, s3_finf0, s3_finf1, s3_achn0, s3_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s3_dM_mean)
analysis_stats(s3_benefits_mean)

s4_Cin, s4_dM, s4_benefits, s4_dY, s4_costs, s4_nb, s4_Cin_mean, s4_dM_mean, s4_benefits_mean, s4_dY_mean, s4_costs_mean, s4_nb_mean, s4_finf0, s4_finf1, s4_achn0, s4_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s4_dM_mean)
analysis_stats(s4_benefits_mean)

s5_Cin, s5_dM, s5_benefits, s5_dY, s5_costs, s5_nb, s5_Cin_mean, s5_dM_mean, s5_benefits_mean, s5_dY_mean, s5_costs_mean, s5_nb_mean, s5_finf0, s5_finf1, s5_achn0, s5_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s5_dM_mean)
analysis_stats(s5_benefits_mean)
analysis_stats(s5_Cin_mean)

s6_Cin, s6_dM, s6_benefits, s6_dY, s6_costs, s6_nb, s6_Cin_mean, s6_dM_mean, s6_benefits_mean, s6_dY_mean, s6_costs_mean, s6_nb_mean, s6_finf0, s6_finf1, s6_achn0, s6_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s6_dM_mean)
analysis_stats(s6_benefits_mean)
analysis_stats(s6_Cin_mean)

s7_Cin, s7_dM, s7_benefits, s7_dY, s7_costs, s7_nb, s7_Cin_mean, s7_dM_mean, s7_benefits_mean, s7_dY_mean, s7_costs_mean, s7_nb_mean, s7_finf0, s7_finf1, s7_achn0, s7_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s7_dM_mean)
analysis_stats(s7_benefits_mean)
analysis_stats(s7_Cin_mean)

s8_Cin, s8_dM, s8_benefits, s8_dY, s8_costs, s8_nb, s8_Cin_mean, s8_dM_mean, s8_benefits_mean, s8_dY_mean, s8_costs_mean, s8_nb_mean, s8_finf0, s8_finf1, s8_achn0, s8_achn1 = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
analysis_stats(s8_dM_mean)
analysis_stats(s8_benefits_mean)
