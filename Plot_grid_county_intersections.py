# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 12:51:34 2024

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

#%%

county = init_County()
grid = init_AQ_Grids()

grid['GRID_KEY'] = grid['GRID_KEY'].apply(lambda x: str(x))  # Convert tuples to string if not already

# Your list of tuples to filter by
grid_list = [(17, 6), (17, 7), (18, 6), (18, 7)]
grid_list_str = [str(item) for item in grid_list]  # Convert list of tuples to list of strings

# Filtering
Target_Grid = grid[grid['GRID_KEY'].isin(grid_list_str)]

Target_County = county[ county['FIPS'] == 'F01055']

#%%

base = grid.plot(color='lightgrey', figsize=(10, 10), edgecolor='k')

# Plot 'Target_Grid' on top with a darker color
Target_Grid.plot(ax=base, color='darkgrey', edgecolor='k')

# Plot 'Target_County' on top with specified alpha and contrasting color
Target_County.plot(ax=base, color='blue', alpha=0.5, edgecolor='k')

plt.show()

base = Target_Grid.plot(color='lightgrey', figsize=(10, 10), edgecolor='k')

# Plot 'Target_County' on top with specified alpha and contrasting color
Target_County.plot(ax=base, color='blue', alpha=0.5, edgecolor='k')

plt.show()

#%%

fig, axes = plt.subplots(1, 2, figsize=(20, 10))  # Create 1 row, 2 columns of subplots

# Plot the first set of GeoDataFrames on the first subplot
base_left = grid.plot(ax=axes[0], color='lightgrey', edgecolor='k')
Target_Grid.plot(ax=axes[0], color='darkgrey', edgecolor='k')
Target_County.plot(ax=axes[0], color='blue', alpha=0.5, edgecolor='k')

# Set title for the first subplot
axes[0].set_title('Grid and Target Layers')

# Plot the second set of GeoDataFrames on the second subplot
base_right = Target_Grid.plot(ax=axes[1], color='lightgrey', figsize=(10, 10), edgecolor='k')
Target_County.plot(ax=axes[1], color='blue', alpha=0.5, edgecolor='k')

# Set title for the second subplot
axes[1].set_title('Target Layers')

plt.show()