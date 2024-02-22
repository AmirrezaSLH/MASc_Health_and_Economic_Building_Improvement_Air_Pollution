# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:02 2024

@author: asalehi
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import numpy as np
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

#%%

#This version converts grid keys to string before exporting.
def County_to_Grid_percentage_mapping(County_gdf = init_County(),AQ_Grid_gdf = init_AQ_Grids()):
    
    county_columns_to_keep = ['FIPS', 'geometry']
    ci = County_gdf[county_columns_to_keep].copy()
    
    grid_columns_to_keep = ['GRID_KEY', 'geometry']
    AQgrid = AQ_Grid_gdf[grid_columns_to_keep].copy()
    
    ci.crs = "EPSG:4326"
    AQgrid.crs = "EPSG:4326"
    
    ci = ci.to_crs("EPSG:5070")
    AQgrid = AQgrid.to_crs("EPSG:5070")
    
    ci['Country_Area'] = ci.area
    
    joined = gpd.overlay(AQgrid, ci, how = 'intersection')
    # Calculate Joined area
    joined['Intersect_Area'] = joined.area
    # Caclulcate percentage of DMA area in each grid cell
    joined['Percentage_Area'] = (joined['Intersect_Area']/joined['Country_Area'])
    
    joined_copy = joined.copy()
    
    County_to_Grid_percentage_mapping_dict = {}
    County_List = joined['FIPS'].unique().tolist()
    print(len(County_List))
    i = 0
    for c in County_List:
        County_to_Grid_percentage_mapping_dict[c] = {}
        i +=1
        print(i)
        for IDX, row in joined_copy.iterrows():
            # Check if the centroid of the county is within the current grid cell
            if row['FIPS'] == c:
                GRID_KEY = str(row['GRID_KEY'])
                County_to_Grid_percentage_mapping_dict[c][ GRID_KEY ] = row['Percentage_Area']
                #joined_copy = joined_copy.drop(joined_copy.index[IDX])
    
    return joined, County_to_Grid_percentage_mapping_dict

#%%

test, test2 = County_to_Grid_percentage_mapping()
test.plot()

empty, test3 = County_to_Grid_percentage_mapping()
# Define the file name
out_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Github_Adaptation\MASc_Waterloo_Adaptation_Buildings\Data\\'
filename = 'County_to_Grid_percentage_mapping.json'

# Write the dictionary to a file
with open(out_dir + filename, 'w') as file:
    json.dump(test3, file, indent=4)
    
#%%

# Example tuple
my_tuple = (1, 2)

# Convert tuple to string
tuple_str = str(my_tuple)
print(tuple_str)
# or for custom formatting:
tuple_str = f'({my_tuple[0]},{my_tuple[1]})'

print(tuple_str)  # Output: '(1,2)'
