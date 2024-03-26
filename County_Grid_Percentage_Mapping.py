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
    AQ_Grid_gdf['GRID_KEY'] = [str((col, row)) for col, row in zip(AQ_Grid_gdf['COL'], AQ_Grid_gdf['ROW'])]
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

#test, test2 = County_to_Grid_percentage_mapping()
#test.plot()

empty, test3 = County_to_Grid_percentage_mapping()
# Define the file name
out_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Github_Adaptation\MASc_Waterloo_Adaptation_Buildings\Data\\'
filename = 'County_to_Grid_percentage_mapping.json'

# Write the dictionary to a file
with open(out_dir + filename, 'w') as file:
    json.dump(test3, file, indent=4)

#%% Optimized version

def County_to_Grid_percentage_mapping_optimizied(County_gdf = init_County(),AQ_Grid_gdf = init_AQ_Grids()):
    """
    Efficiently calculates the percentage of each county's area that intersects
    with each AQ grid cell.
    
    Parameters:
    - County_gdf: GeoDataFrame containing county geometries.
    - AQ_Grid_gdf: GeoDataFrame containing AQ grid geometries.
    
    Returns:
    - Tuple: (joined GeoDataFrame with intersection details, dictionary mapping county FIPS to grid cell percentages)
    """
    County_gdf.crs = "EPSG:4326"
    AQ_Grid_gdf.crs = "EPSG:4326"
    # Ensure CRS is consistent and in a suitable projection for area calculations
    if County_gdf.crs != "EPSG:5070":
        County_gdf = County_gdf.to_crs("EPSG:5070")
    if AQ_Grid_gdf.crs != "EPSG:5070":
        AQ_Grid_gdf = AQ_Grid_gdf.to_crs("EPSG:5070")
    
    # Calculate area of each county
    County_gdf['Country_Area'] = County_gdf.area
    
    # Perform spatial intersection
    joined = gpd.overlay(AQ_Grid_gdf, County_gdf, how='intersection')
    joined['Intersect_Area'] = joined.area  # Calculate area of the intersection
    joined['Percentage_Area'] = joined['Intersect_Area'] / joined['Country_Area']
    
    # Create the mapping dictionary without iterating over DataFrame rows
    mapping_dict = (joined.groupby('FIPS')
                           .apply(lambda x: dict(zip( x['GRID_KEY'].apply(lambda gk: str(gk)), x['Percentage_Area'])))
                           .to_dict())

    return joined, mapping_dict
#%%

empty_optimized, test_optimized = County_to_Grid_percentage_mapping_optimizied()

if test_optimized == test3:
    print("The dictionaries are identical.")
else:
    print("The dictionaries are not identical.")
#%%

# Example tuple
my_tuple = (1, 2)

# Convert tuple to string
tuple_str = str(my_tuple)
print(tuple_str)
# or for custom formatting:
tuple_str = f'({my_tuple[0]},{my_tuple[1]})'

print(tuple_str)  # Output: '(1,2)'

#%%
#This version converts grid keys to string before exporting.
def Grid_to_County_percentage_mapping(County_gdf = init_County(),AQ_Grid_gdf = init_AQ_Grids()):
    
    county_columns_to_keep = ['FIPS', 'geometry']
    ci = County_gdf[county_columns_to_keep].copy()
    
    grid_columns_to_keep = ['GRID_KEY', 'geometry']
    AQgrid = AQ_Grid_gdf[grid_columns_to_keep].copy()
    
    ci.crs = "EPSG:4326"
    AQgrid.crs = "EPSG:4326"
    
    ci = ci.to_crs("EPSG:5070")
    AQgrid = AQgrid.to_crs("EPSG:5070")
    
    ci['Country_Area'] = ci.area
    AQgrid['Grid_Area'] = AQgrid.area
    
    joined = gpd.overlay(ci, AQgrid, how = 'intersection')
    # Calculate Joined area
    joined['Intersect_Area'] = joined.area
    # Caclulcate percentage of DMA area in each grid cell
    joined['Percentage_Area'] = (joined['Intersect_Area']/joined['Grid_Area'])
    
    joined_copy = joined.copy()
    
    Grid_to_County_percentage_mapping_dict = {}
    Grid_List = joined['GRID_KEY'].unique().tolist()
    print(len(Grid_List))
    i = 0
    for g in Grid_List:
        Grid_to_County_percentage_mapping_dict[g] = {}
        i +=1
        print(i)
        for IDX, row in joined_copy.iterrows():
            # Check if the centroid of the county is within the current grid cell
            if row['GRID_KEY'] == g:
                c = str(row['FIPS'])
                Grid_to_County_percentage_mapping_dict[g][ c ] = row['Percentage_Area']
                #joined_copy = joined_copy.drop(joined_copy.index[IDX])
    
    return joined, Grid_to_County_percentage_mapping_dict

#%%

def Grid_to_County_percentage_mapping_optimizied(County_gdf = init_County(),AQ_Grid_gdf = init_AQ_Grids()):
    """
    Efficiently calculates the percentage of each county's area that intersects
    with each AQ grid cell.
    
    Parameters:
    - County_gdf: GeoDataFrame containing county geometries.
    - AQ_Grid_gdf: GeoDataFrame containing AQ grid geometries.
    
    Returns:
    - Tuple: (joined GeoDataFrame with intersection details, dictionary mapping county FIPS to grid cell percentages)
    """
    County_gdf.crs = "EPSG:4326"
    AQ_Grid_gdf.crs = "EPSG:4326"
    # Ensure CRS is consistent and in a suitable projection for area calculations
    if County_gdf.crs != "EPSG:5070":
        County_gdf = County_gdf.to_crs("EPSG:5070")
    if AQ_Grid_gdf.crs != "EPSG:5070":
        AQ_Grid_gdf = AQ_Grid_gdf.to_crs("EPSG:5070")
    
    # Calculate area of each county
    County_gdf['Country_Area'] = County_gdf.area
    AQ_Grid_gdf['AQ_Area'] = AQ_Grid_gdf.area

    # Perform spatial intersection
    joined = gpd.overlay(County_gdf, AQ_Grid_gdf, how='intersection')
    joined['Intersect_Area'] = joined.area  # Calculate area of the intersection
    joined['Percentage_Area'] = joined['Intersect_Area'] / joined['AQ_Area']
    
    # Create the mapping dictionary without iterating over DataFrame rows
    mapping_dict = (joined.groupby('GRID_KEY')
                           .apply(lambda x: dict(zip( x['FIPS'], x['Percentage_Area'])))
                           .to_dict())

    return joined, mapping_dict

#%%

empty_reverse, test_reverse = Grid_to_County_percentage_mapping_optimizied()

out_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Github_Adaptation\General_Processed_Data\\'
filename = 'Spatial_County_contents_in_Grid_Overlap_Percentages.json'

# Write the dictionary to a file
with open(out_dir + filename, 'w') as file:
    json.dump(test_reverse, file, indent=4)
    
dt = test_reverse[(8, 7)]
s = 0
for j in test_reverse.keys():
    for i in test_reverse[j].values():
        s += i
    print(s)
    s=0