# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:29:11 2024

@author: asalehi
"""

#%%
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

#%%%

import main
import plot_functions

#%%

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

def calculate_spatial_mean( data_list):
    
    """
    Calculate the mean of values across a list of dictionaries,
    assuming each dictionary has the same keys.

    Parameters:
    - data_list: List of dictionaries with numeric values and the same keys.

    Returns:
    - dict: Dictionary with the mean values computed across the dictionaries for each key.
    """
    data_mean = {}
    grid_list = data_list[0].keys()
    
    for g in grid_list:
        # Collect the values for each grid from all dictionaries
        values = [data[g] for data in data_list]
        
        # Calculate mean using numpy for efficiency
        data_mean[g] = np.mean(values)
    
    return data_mean
            
            
    
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

def generate_maps(dCin, dY, benefit, cost, nb, scenario ='' ):
    
    dY_mean = calculate_spatial_mean(dY)
    
    pop_dict = main.init_population()
    
    benefit_mean = calculate_spatial_mean(benefit)
    benefit_capita = {key : value/pop_dict[key] for key, value in benefit_mean.items()}
    
    cost_mean = calculate_spatial_mean(cost)
    cost_capita = {key : value/pop_dict[key] for key, value in cost_mean.items()}
    
    nb_mean = calculate_spatial_mean(nb)
    nb_capita = {key : value/pop_dict[key] for key, value in nb_mean.items()}
    
    
    
    #cost_capita_mean = calculate_spatial_mean(cost_capita)
    #nb_capita_mean = calculate_spatial_mean(nb_capita)
    dCin_mean = calculate_spatial_mean(dCin)
    
    aq_grid_gdf = main.init_AQ_grids()
    legend_ticks = [5, 15, 25, 35, 45, 55]
    plot_functions.create_contour_plot_dy(aq_grid_gdf, dY_mean, 'dY', scenario + 'Mortality Incidence Reduction', 'Per 100,000 people', color ='blue')
    plot_functions.create_contour_plot_dC(aq_grid_gdf, dCin_mean, 'dC', scenario +'Average Residence $PM_2.5$ Reduction', r'$\mu g/m^3$', color ='blue')
    
    plot_functions.create_contour_plot(aq_grid_gdf, benefit_capita, 'bc', scenario +'Benefit per Capita', 'Billion USD', color ='blue')
    plot_functions.create_contour_plot(aq_grid_gdf, cost_capita, 'cc', scenario +'Cost per Capita', 'Billion USD', color ='blue')
    plot_functions.create_contour_plot(aq_grid_gdf, nb_capita, 'nbc', scenario +'Net Benefit per Capita', 'Billion USD', color ='red_blue')
    
    return 0

def generate_box_plots(dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national, n, scenario_list):
    
    pop_dict = main.init_population()
    print(dY_national)
    plot_functions.plot_boxplots(dY_national, 'National Annual Mortality Incidence Rate Reduction', scenario_list,'per 100,000')
    plot_functions.plot_boxplots(dM_national, 'National Annual Avoided Mortality', scenario_list,'People')
    plot_functions.plot_boxplots(benefits_national, 'National Annual Total Benefits', scenario_list,'Billion USD')
    plot_functions.plot_boxplots(costs_national, 'National Annual Total Costs', scenario_list,'Billion USD')
    plot_functions.plot_boxplots(net_benefits_national, 'National Annual Total Net Benefits', scenario_list,'Billion USD')
    
    #Total Cost Benefit 
    cost_benefit = [benefits_national, net_benefits_national, costs_national]  # List of datasets
    cost_benefit_group_labels  = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    plot_functions.plot_grouped_boxplots(cost_benefit, 'National Annual Cost Benefit Analysis', cost_benefit_group_labels, scenario_list, 'Billion USD')
    
    '''
    #Individual Cost Benefit 
    net_benefit_ind = (net_benefits_national / sum(pop_dict.values())) * 1000000000
    benefit_ind = (benefits_national / sum(pop_dict.values())) * 1000000000
    cost_ind = (costs_national / sum(pop_dict.values())) * 1000000000
    
    cost_benefit_ind = [benefit_ind, net_benefit_ind, cost_ind]  # List of datasets
    cost_benefit_ind_xlabels = ['Benefits', 'Net Benefits', 'Costs']  # Labels for each box plot
    plot_boxplots(cost_benefit_ind, 'National Annual Cost Benefit Analysis per person', cost_benefit_ind_xlabels, 'USD', colors=['skyblue', 'green', 'red'])
    '''
    plot_functions.plot_boxplots(dCin_national, 'National Annual $PM_2.5$ Reduction in Indoor Concentrations', scenario_list, r'$\mu g/m^3$')
    
    #plot_histogram(dY_national, 'National Annual Mortality Incidence Rate Reduction', 'per 100,000', 'count')

    return dY_national

def scenario_UC1(re_risk, int_func, itr, hf = 'HF_3'):
    dCin, dM, benefits, dY, costs, net_benefits, finf0, finf1, achn0, achn1 = main.run_MCS(  intervention_function = int_func, rr_beta = re_risk , iterations = itr, HF = hf)
    dCin_national, dM_national, dY_national, costs_national, benefits_national, net_benefits_national = aggregate_results( dCin, dM, benefits, dY, costs, net_benefits, itr)
    
    return dCin, dM, benefits, dY, costs, net_benefits, dCin_national, dM_national, benefits_national, dY_national, costs_national, net_benefits_national, finf0, finf1, achn0, achn1

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

def reduction_energy_star( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.75
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.75
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.75
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention


def reduction_20( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.8
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.8
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.8
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention

def reduction_40( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.6
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.6
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.6
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention

def reduction_60( ach50_segment_dict, GRID_KEY, climatezone_grid_map, P_d = 0.97, K_d = 0.37 ):
    
    ach50_segment_dict['SF_V3'] = ach50_segment_dict['SF_V3'] * 0.4
    ach50_segment_dict['SF_V2'] = ach50_segment_dict['SF_V2'] * 0.4
    ach50_segment_dict['SF_V1'] = ach50_segment_dict['SF_V1'] * 0.4
    
    leakage_intervention = {key: main.ACH50_to_Finf(value, P = P_d, K = K_d) for key, value in ach50_segment_dict.items()}
    Finf_intervention = {key: value[0] for key, value in leakage_intervention.items()}
    ACH_intervention = {key: value[1] for key, value in leakage_intervention.items()}
    
    return Finf_intervention, ACH_intervention

#%%

rR = 0.005826891
n=10

sdCin1, sdM1, sbenefits1, sdY1, scosts1, snet_benefits1, delta_mean_C_in1, dMort1, benefit1, dY1, C1, nb1, _,_,_,_ = scenario_UC1( re_risk = rR, int_func = code_comply_all, itr = n, hf = 'HF_3')
sdCin2, sdM2, sbenefits2, sdY2, scosts2, snet_benefits2, delta_mean_C_in2, dMort2, benefit2, dY2, C2, nb2, _,_,_,_ = scenario_UC1( re_risk = rR, int_func = reduction_20, itr = n, hf = 'HF_3')
sdCin3, sdM3, sbenefits3, sdY3, scosts3, snet_benefits3, delta_mean_C_in3, dMort3, benefit3, dY3, C3, nb3, _,_,_,_ = scenario_UC1( re_risk = rR, int_func = reduction_energy_star, itr = n, hf = 'HF_3')
sdCin4, sdM4, sbenefits4, sdY4, scosts4, snet_benefits4, delta_mean_C_in4, dMort4, benefit4, dY4, C4, nb4, _,_,_,_ = scenario_UC1( re_risk = rR, int_func = reduction_40, itr = n, hf = 'HF_3')
sdCin5, sdM5, sbenefits5, sdY5, scosts5, snet_benefits5, delta_mean_C_in5, dMort5, benefit5, dY5, C5, nb5, _,_,_,_ = scenario_UC1( re_risk = rR, int_func = reduction_60, itr = n, hf = 'HF_3')


analysis_stats(benefit1)
analysis_stats(benefit2)
analysis_stats(benefit3)
analysis_stats(nb3)

delta_mean_C_in = [delta_mean_C_in2, delta_mean_C_in3, delta_mean_C_in4, delta_mean_C_in5, delta_mean_C_in1]
dMort = [ dMort2, dMort3, dMort4, dMort5, dMort1]
benefit = [ benefit2, benefit3, benefit4, benefit5, benefit1]
dY = [dY2, dY3, dY4, dY5, dY1]
C = [ C2, C3, C4, C5, C1]
nb = [nb2, nb3, nb4, nb5, nb1]

x = [ '20% Reduction', 'Energy Star', '40% Reduction', '60% Reduction', 'Comply IECC']
testtest = generate_box_plots(dCin_national =delta_mean_C_in, dM_national=dMort, dY_national=dY, costs_national=C, benefits_national=benefit, net_benefits_national=nb, n=n, scenario_list=x)

generate_maps( dCin = sdCin1, dY = sdY1 , benefit = sbenefits1, cost = scosts1, nb = snet_benefits1 )
generate_maps( dCin = sdCin2, dY = sdY2 , benefit = sbenefits2, cost = scosts2, nb = snet_benefits2 )

generate_maps( dCin = sdCin3, dY = sdY3 , benefit = sbenefits3, cost = scosts3, nb = snet_benefits3, scenario = 'Energy Star: ' )
generate_maps( dCin = sdCin4, dY = sdY4 , benefit = sbenefits4, cost = scosts4, nb = snet_benefits4 )

generate_maps( dCin = sdCin5, dY = sdY5 , benefit = sbenefits5, cost = scosts5, nb = snet_benefits5 )

aq_grid_gdf = main.init_AQ_grids()
nb_mean = calculate_spatial_mean(snet_benefits1)
aq_grid_gdf['nb'] = aq_grid_gdf['GRID_KEY'].map(nb_mean)

aq_grid_gdf = aq_grid_gdf.dropna(subset=['nb'])
aq_grid_gdf['nb'] = -aq_grid_gdf['nb']
