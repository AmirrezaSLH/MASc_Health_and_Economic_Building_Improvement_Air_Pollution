# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 17:14:25 2024

@author: asalehi
"""

import pandas as pd
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import json

#%%

def init_AQ_Grids( dir = 'Data/AQgrid.gpkg'):
    #This Function Loads the AQ Grids
    AQ_Grid_gdf = gpd.read_file(dir)
    AQ_Grid_gdf['GRID_KEY'] = [(col, row) for col, row in zip(AQ_Grid_gdf['COL'], AQ_Grid_gdf['ROW'])]

    return AQ_Grid_gdf

def init_County( dir = 'Data/County_Main_Land.gpkg'):
    #This Function Loads the Counties
    County_gdf = gpd.read_file(dir)
    return County_gdf

def init_Buildings_Stock( dir = 'Data/Buildings_Stock.csv'):
    #This Function Loads the Buildings Stock
    Buildings_Stock_df = pd.read_csv(dir)
    return Buildings_Stock_df


def init_Country_to_Grid_Percentage_Mapping(  dir = 'Data/County_to_Grid_percentage_mapping.json' ):
    with open(dir, 'r') as file:
        Country_to_Grid_Percentage_Mapping = json.load(file)
    
    return Country_to_Grid_Percentage_Mapping
    
#%%
'''
def Buildings_random_sample(df, mapping, grid):
    
    proportions = mapping.values().tolist()
    gvalue = mapping[grid]
    gindex = proportions.index(gvalue)
    print(sum(proportions))
    #assert sum(proportions) <= 1, "Sum of proportions must be less than or equal to 1"
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    
    # Calculate the indices for splits based on proportions
    total_len = len(df)
    indices = np.cumsum([int(p * total_len) for p in proportions])
    
    # Split the DataFrame into parts
    parts = []
    start_idx = 0
    for end_idx in indices:
        parts.append(df_shuffled.iloc[start_idx:end_idx])
        start_idx = end_idx
    
    # If the sum of proportions is less than 1, add the remaining data as an additional part
    #if sum(proportions) < 1:
    #    parts.append(df_shuffled.iloc[start_idx:])
    
    return parts[gindex]
'''

def Buildings_random_sample(df, mapping, grid):
    sampled_df = df.sample(frac=0.2)
    
    return sampled_df

def ACH50_Occupancy_Average( Target_Stock):
    Target_Stock['ACH50_occupants'] = Target_Stock['ACH50'] * Target_Stock['occupants']
    Occupants_sum = Target_Stock['occupants'].sum()
    ACH50_mean = Target_Stock['ACH50_occupants'].sum() / Occupants_sum
    
    return ACH50_mean

def Buildings_to_Grid_Sample(Buildings_Stock_df = init_Buildings_Stock(), County_to_Grid_mapping_dict = init_Country_to_Grid_Percentage_Mapping(), Grid_gdf = init_AQ_Grids(), County_gdf = init_County() ):
    
    county_list = County_gdf['FIPS'].unique().tolist()
    bt_list = Buildings_Stock_df['type'].unique().tolist()
    grid_list = Grid_gdf['GRID_KEY'].unique().tolist() 
    #Splitting buildings based on their county
    county_building_type_dict = {}
    for BT in bt_list:
        county_building_type_dict[BT] = {}
        BT_building_df = Buildings_Stock_df[ Buildings_Stock_df['type']  == BT]
        for c in county_list:
            county_building_df = BT_building_df[ BT_building_df['FIPS']  == c]
            county_building_type_dict[BT][c] = county_building_df
    print("initiate")
    list_ACH50 = []
    Buildings_for_ACH50_dict = {}
    for BT in bt_list:
        Buildings_for_ACH50_dict[BT] = {}
        for g in grid_list:
            list_ACH50 = []
            for i in range(1000):
                print(BT, ' : ',g, ' : ', i)
                Buildings_in_Grid = pd.DataFrame()
                for c in county_list:
                    Grid_to_Percentage = County_to_Grid_mapping_dict[c]
                    str_g = str(g)
                    if str_g in Grid_to_Percentage.keys():
                        #print("YES")
                        if len(Grid_to_Percentage.keys()) == 1:
                            #print("YES_2")
                            Buildings_in_Grid = pd.concat([Buildings_in_Grid, county_building_type_dict[BT][c]], ignore_index=True)
                            
                        else:
                            
                            Building_Sample = Buildings_random_sample( county_building_type_dict[BT][c], Grid_to_Percentage, str_g)
                            
                            Buildings_in_Grid = pd.concat([Buildings_in_Grid, Building_Sample], ignore_index=True)
                            
                #print(Buildings_in_Grid)
                #print(BT, ' : ',g)
                ACH50_mean = ACH50_Occupancy_Average( Buildings_in_Grid )
                list_ACH50.append(ACH50_mean)
            str_g = str(g)
            Buildings_for_ACH50_dict[BT][str_g] = list_ACH50  
    
    return county_building_type_dict, Buildings_for_ACH50_dict

#SSss

'''
def Buildings_to_Grid_Sample(Buildings_Stock_df = init_Buildings_Stock(), County_to_Grid_mapping_dict = init_Country_to_Grid_Percentage_Mapping(), Grid_gdf = init_AQ_Grids(), County_gdf = init_County() ):
    
    county_list = County_gdf['FIPS'].unique().tolist()
    bt_list = Buildings_Stock_df['type'].unique().tolist()
    grid_list = Grid_gdf['GRID_KEY'].unique().tolist() 
    #Splitting buildings based on their county
    county_building_type_dict = {}
    for BT in bt_list:
        county_building_type_dict[BT] = {}
        BT_building_df = Buildings_Stock_df[ Buildings_Stock_df['type']  == BT]
        for c in county_list:
            county_building_df = BT_building_df[ BT_building_df['FIPS']  == c]
            county_building_type_dict[BT][c] = county_building_df
    print("initiate")
    list_ACH50 = []
    for i in range(10):
        print(i)
        Buildings_for_ACH50_dict = {}
        for BT in bt_list:
            Buildings_for_ACH50_dict[BT] = {}
            for g in grid_list:
                Buildings_in_Grid = pd.DataFrame()
                for c in county_list:
                    Grid_to_Percentage = County_to_Grid_mapping_dict[c]
                    str_g = str(g)
                    if str_g in Grid_to_Percentage.keys():
                        #print("YES")
                        if len(Grid_to_Percentage.keys()) == 1:
                            #print("YES_2")
                            Buildings_in_Grid = pd.concat([Buildings_in_Grid, county_building_type_dict[BT][c]], ignore_index=True)
                            
                        else:
                            
                            Building_Sample = Buildings_random_sample( county_building_type_dict[BT][c], Grid_to_Percentage, str_g)
                            
                            Buildings_in_Grid = pd.concat([Buildings_in_Grid, Building_Sample], ignore_index=True)
                            
                #print(Buildings_in_Grid)
                #print(BT, ' : ',g)
                ACH50_mean = ACH50_Occupancy_Average( Buildings_in_Grid )
                
                str_g = str(g)
                Buildings_for_ACH50_dict[BT][str_g] = ACH50_mean  
        list_ACH50.append(Buildings_for_ACH50_dict)
    
    return county_building_type_dict, Buildings_for_ACH50_dict, list_ACH50
'''

def Exctract_distribution_plot( Buildings_for_ACH50_dict, Buildings_Stock_df = init_Buildings_Stock(), Grid_gdf = init_AQ_Grids()):
    bt_list = Buildings_Stock_df['type'].unique().tolist()
    grid_list = Grid_gdf['GRID_KEY'].unique().tolist() 
    
    for g in grid_list:
        str_g = str(g)
        plt.hist(Buildings_for_ACH50_dict['Single-Family Detached'][str_g])
        plt.title(str_g+ ' - Single-Family Detached ACH 50 distribution')
        plt.show()
    return
                
    
    
#%%

county_grid_map = init_Country_to_Grid_Percentage_Mapping()
county_grid_map.keys()
grid_test = init_AQ_Grids()
test2 = Buildings_to_Grid_Sample()
test3 = Buildings_to_Grid_Sample()

cbt, bfa, lbfa = Buildings_to_Grid_Sample()
cbt2, bfa2 = Buildings_to_Grid_Sample()
cbt3, bfa3 = Buildings_to_Grid_Sample()
cbt1000, bfa1000 = Buildings_to_Grid_Sample()

out_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Github_Adaptation\MASc_Waterloo_Adaptation_Buildings\Data\\'
filename = 'ACH50_Grid_1000sim.json'

# Write the dictionary to a file
with open(out_dir + filename, 'w') as file:
    json.dump(bfa1000, file, indent=4)
#%%

Exctract_distribution_plot(bfa1000)
