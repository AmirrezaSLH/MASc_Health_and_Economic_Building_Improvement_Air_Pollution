# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:11:08 2024

@author: asalehi
"""

import numpy as np
import geopandas as gpd

import json

from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
from matplotlib.patches import PathPatch

from shapely.geometry import Polygon

from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

import pandas as pd
import math
#%%

def count_digits(num):
    return len(str(abs(num)))

def create_contour_plot(gdf, data, tc, title='', legend_label='', legend_ticks=None, color = 'red'):
    """
    Create a contour plot of Air Quality Grid data with PM concentrations.

    Parameters:
    - aq_grid_gdf: GeoDataFrame with Air Quality Grid data.
    - pm_data: Dictionary with PM2.5 concentration data.
    - target_column: Column name of the value to plot.
    - contour_levels: Number of levels in the contour plot.
    - cmap_name: Name of the colormap to use.
    - linewidth: Width of the lines on the US map.
    - figsize: Size of the matplotlib figure.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    #gdf[tc] = -gdf[tc]
    gdf.plot(column = tc, legend = True)
    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]  # Adjust grid resolution as needed

    # Interpolate the data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')


    # Load a base map of the US
    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter out Alaska, Hawaii, and other territories to keep only the mainland US
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]

    # unary_union creates a single geometry out of MultiPolygon
    us_boundary = unary_union(us_map.geometry)
    # If the boundary is a MultiPolygon (which it will be if there are islands), we will combine them
    if isinstance(us_boundary, MultiPolygon):
        us_boundary = us_boundary.simplify(0.05)  # You might need to adjust the tolerance to your specific case
        exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms if poly.is_valid and not poly.is_empty]
    else:
        exterior_coords = [np.array(us_boundary.exterior.coords)]

    # Flatten the list of arrays into a single array and create a Path object
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    # Create the mask
    inside_us = np.array([
        us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)
    
    # Apply the mask to your grid_z
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    # Plot the base map
    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Plot the contour map with the masked grid
    if color == 'custom':
        
        #range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])
        range_cmap = LinearSegmentedColormap.from_list('white_to_lightblue', ['white', '#9999FF'])
    elif color == 'green':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'green'])
    elif color == 'blue':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'blue'])
    elif color == 'red':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])

    if legend_ticks is None:
        contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    else: 
        c_min = min(legend_ticks)
        c_max = max(legend_ticks)
        contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(c_min, c_max, 1000), cmap=range_cmap, alpha=0.9)
    
    #fig.colorbar(contour, ax=ax, label='Net Benefits')

    # Set the plot limits to the bounds of the contiguous US, excluding Alaska and Hawaii
    # You may need to manually set these limits to fit the mainland US appropriately
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    
    #cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20, label=legend_label)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)  # You can adjust the fontsize as needed.

    if title:
        ax.set_title(title, fontsize=22)

    # Removing frame around the map
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Customizing color bar ticks
    if legend_ticks is not None:
        print(legend_ticks)
        cbar.set_ticks(legend_ticks)
        cbar.set_ticklabels([str(tick) for tick in legend_ticks])
    else:
        tick_start = gdf[tc].min()
        tick_end = gdf[tc].max()
        ticks = np.linspace(tick_start, tick_end, num=5, endpoint=True)  # Generates 5 evenly spaced ticks
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
        
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)  
    # Show the plot
    plt.show()

#%%
def create_contour_plot_negetive(gdf, data, tc, title='', legend_label='', legend_ticks=None, color = 'red'):
    """
    Create a contour plot of Air Quality Grid data with PM concentrations.

    Parameters:
    - aq_grid_gdf: GeoDataFrame with Air Quality Grid data.
    - pm_data: Dictionary with PM2.5 concentration data.
    - target_column: Column name of the value to plot.
    - contour_levels: Number of levels in the contour plot.
    - cmap_name: Name of the colormap to use.
    - linewidth: Width of the lines on the US map.
    - figsize: Size of the matplotlib figure.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    #gdf[tc] = -gdf[tc]
    gdf.plot(column = tc, legend = True)
    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]  # Adjust grid resolution as needed

    # Interpolate the data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')


    # Load a base map of the US
    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter out Alaska, Hawaii, and other territories to keep only the mainland US
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]

    # unary_union creates a single geometry out of MultiPolygon
    us_boundary = unary_union(us_map.geometry)
    # If the boundary is a MultiPolygon (which it will be if there are islands), we will combine them
    if isinstance(us_boundary, MultiPolygon):
        us_boundary = us_boundary.simplify(0.05)  # You might need to adjust the tolerance to your specific case
        exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms if poly.is_valid and not poly.is_empty]
    else:
        exterior_coords = [np.array(us_boundary.exterior.coords)]

    # Flatten the list of arrays into a single array and create a Path object
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    # Create the mask
    inside_us = np.array([
        us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)
    
    # Apply the mask to your grid_z
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    # Plot the base map
    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Plot the contour map with the masked grid

    #negative_count = (gdf[tc] < 0).sum()
    #turning_point = negative_count/len(gdf[tc])
    max_val = gdf[tc].max()
    min_val = gdf[tc].min()
    turning_point = (0 - min_val) / (max_val - min_val)
    cdict = {
    'red':   ((0.0, 1.0, 1.0),   # Red at the lowest
              (0.5, 1.0, 1.0),   # White at the middle
              (1.0, 0.0, 0.0)),  # Blue at the highest
        
    'green': ((0.0, 0.0, 0.0),   # Green off for the extremes
              (0.5, 1.0, 1.0),   # White at the middle
              (1.0, 0.0, 0.0)),
        
    'blue':  ((0.0, 0.0, 0.0),   # Blue off at the lowest
              (0.5, 1.0, 1.0),   # White at the middle
              (1.0, 1.0, 1.0))   # Blue at the highest
    }
    range_cmap = LinearSegmentedColormap('white_to_red', cdict)

    #contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(-z.max(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    #fig.colorbar(contour, ax=ax, label='Net Benefits')

    # Set the plot limits to the bounds of the contiguous US, excluding Alaska and Hawaii
    # You may need to manually set these limits to fit the mainland US appropriately
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    
    #cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20, label=legend_label)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)  # You can adjust the fontsize as needed.

    # Customizing color bar ticks
    if legend_ticks is not None:
        cbar.set_ticks(legend_ticks)
        cbar.set_ticklabels([str(tick) for tick in legend_ticks])
    
    # Customizing plot title
    if title:
        ax.set_title(title, fontsize=22)

    # Removing frame around the map
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Customizing the axes ticks
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Removing coordinate numbers (tick labels)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # Dynamically set legend ticks to integer values within a reasonable range
    
    
    tick_start = 0
    #tick_start = - gdf[tc].max()
    tick_end = gdf[tc].max()
    
    #n = tick_end / 4
    #n= math.floor(n)
    #print('NFIRST: ', n, ' : and : ', count_digits(n))
    #n = (n // ( 10 ** (count_digits(n)-1) )) * ( 10 ** (count_digits(n)-1) )
    #print('NNNNN: ', n)
    #ticks = [ -4*n,-3*n, -2*n, -1*n ,0, n, 2*n, 3*n, 4*n ]
    ticks = np.linspace(tick_start, tick_end, num=4, endpoint=True)  # Generates 9 reasonable ticks 0 in the middle
    ticks = [ math.floor(n) for n in ticks]
    index_closest_to_zero = np.argmin(np.abs(ticks))
    # Replace the closest value with zero
    ticks[index_closest_to_zero] = 0
    ticks = np.delete(ticks, index_closest_to_zero)
    

    print(ticks)
    ticks = [ ( n // ( 10 ** (count_digits(n)-1) )) * ( 10 ** (count_digits(n)-1) ) for n in ticks ]
    negetive_ticks = [ -nt for nt in ticks]

    new_ticks = []
    new_ticks = negetive_ticks + [0] + ticks
    print(new_ticks)
    
    new_ticks = [ -400, -200, 0 , 200, 400]
    #print(ticks)
    cbar.set_ticks(new_ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in new_ticks])
    
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    # Show the plot
    plt.show()

#%%
def create_contour_plot_dy(gdf, data, tc, title='', legend_label='', legend_ticks=None, color = 'red'):
    """
    Create a contour plot of Air Quality Grid data with PM concentrations.

    Parameters:
    - aq_grid_gdf: GeoDataFrame with Air Quality Grid data.
    - pm_data: Dictionary with PM2.5 concentration data.
    - target_column: Column name of the value to plot.
    - contour_levels: Number of levels in the contour plot.
    - cmap_name: Name of the colormap to use.
    - linewidth: Width of the lines on the US map.
    - figsize: Size of the matplotlib figure.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    gdf[tc] = (gdf[tc] * 100000).round().astype(int)

    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]  # Adjust grid resolution as needed

    # Interpolate the data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')


    # Load a base map of the US
    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter out Alaska, Hawaii, and other territories to keep only the mainland US
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]

    # unary_union creates a single geometry out of MultiPolygon
    us_boundary = unary_union(us_map.geometry)
    # If the boundary is a MultiPolygon (which it will be if there are islands), we will combine them
    if isinstance(us_boundary, MultiPolygon):
        us_boundary = us_boundary.simplify(0.05)  # You might need to adjust the tolerance to your specific case
        exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms if poly.is_valid and not poly.is_empty]
    else:
        exterior_coords = [np.array(us_boundary.exterior.coords)]

    # Flatten the list of arrays into a single array and create a Path object
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    # Create the mask
    inside_us = np.array([
        us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)
    
    # Apply the mask to your grid_z
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    # Plot the base map
    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Plot the contour map with the masked grid
    if color == 'red':
        
        #range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])
        range_cmap = LinearSegmentedColormap.from_list('white_to_lightblue', ['white', '#9999FF'])
    elif color == 'green':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'green'])
    elif color == 'blue':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'blue'])

    if legend_ticks is None:
        contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    else: 
        c_min = min(legend_ticks)
        c_max = max(legend_ticks)
        contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(c_min, c_max, 1000), cmap=range_cmap, alpha=0.9)
    
    #fig.colorbar(contour, ax=ax, label='Net Benefits')

    # Set the plot limits to the bounds of the contiguous US, excluding Alaska and Hawaii
    # You may need to manually set these limits to fit the mainland US appropriately
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    
    #cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20, label=legend_label)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)  # You can adjust the fontsize as needed.

    if title:
        ax.set_title(title, fontsize=22)

    # Removing frame around the map
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Customizing color bar ticks
    if legend_ticks is not None:
        print(legend_ticks)
        cbar.set_ticks(legend_ticks)
        cbar.set_ticklabels([str(tick) for tick in legend_ticks])
    else:
        tick_start = gdf[tc].min()
        tick_end = gdf[tc].max()
        ticks = np.linspace(tick_start, tick_end, num=5, endpoint=True)  # Generates 5 evenly spaced ticks
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
        
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)  
    # Customizing plot title


    # Customizing the axes ticks
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Removing coordinate numbers (tick labels)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # Dynamically set legend ticks to integer values within a reasonable range
    
    
    # Show the plot
    plt.show()

def create_contour_plot_Air_Pol(gdf, data, tc, title='', legend_label='', legend_ticks=None, color = 'red'):
    """
    Create a contour plot of Air Quality Grid data with PM concentrations.

    Parameters:
    - aq_grid_gdf: GeoDataFrame with Air Quality Grid data.
    - pm_data: Dictionary with PM2.5 concentration data.
    - target_column: Column name of the value to plot.
    - contour_levels: Number of levels in the contour plot.
    - cmap_name: Name of the colormap to use.
    - linewidth: Width of the lines on the US map.
    - figsize: Size of the matplotlib figure.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    #gdf[tc] = -gdf[tc]

    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]  # Adjust grid resolution as needed

    # Interpolate the data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')


    # Load a base map of the US
    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter out Alaska, Hawaii, and other territories to keep only the mainland US
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]

    # unary_union creates a single geometry out of MultiPolygon
    us_boundary = unary_union(us_map.geometry)
    # If the boundary is a MultiPolygon (which it will be if there are islands), we will combine them
    if isinstance(us_boundary, MultiPolygon):
        us_boundary = us_boundary.simplify(0.05)  # You might need to adjust the tolerance to your specific case
        exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms if poly.is_valid and not poly.is_empty]
    else:
        exterior_coords = [np.array(us_boundary.exterior.coords)]

    # Flatten the list of arrays into a single array and create a Path object
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    # Create the mask
    inside_us = np.array([
        us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)
    
    # Apply the mask to your grid_z
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    # Plot the base map
    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Plot the contour map with the masked grid
    if color == 'red':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])
    elif color == 'green':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'green'])
    elif color == 'blue':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'blue'])

    contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    #fig.colorbar(contour, ax=ax, label='Net Benefits')

    # Set the plot limits to the bounds of the contiguous US, excluding Alaska and Hawaii
    # You may need to manually set these limits to fit the mainland US appropriately
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    
    #cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20, label=legend_label)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)  # You can adjust the fontsize as needed.

    # Customizing color bar ticks
    if legend_ticks is not None:
        cbar.set_ticks(legend_ticks)
        cbar.set_ticklabels([str(tick) for tick in legend_ticks])
    
    # Customizing plot title
    if title:
        ax.set_title(title, fontsize=22)

    # Removing frame around the map
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Customizing the axes ticks
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Removing coordinate numbers (tick labels)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # Dynamically set legend ticks to integer values within a reasonable range
    
    tick_start = gdf[tc].min()
    tick_end = gdf[tc].max()
    ticks = np.linspace(tick_start, tick_end, num=5, endpoint=True)  # Generates 5 evenly spaced ticks
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
    
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    # Show the plot
    plt.show()
    
def create_contour_plot_dC(gdf, data, tc, title='', legend_label='', legend_ticks=None, color = 'red'):
    """
    Create a contour plot of Air Quality Grid data with PM concentrations.

    Parameters:
    - aq_grid_gdf: GeoDataFrame with Air Quality Grid data.
    - pm_data: Dictionary with PM2.5 concentration data.
    - target_column: Column name of the value to plot.
    - contour_levels: Number of levels in the contour plot.
    - cmap_name: Name of the colormap to use.
    - linewidth: Width of the lines on the US map.
    - figsize: Size of the matplotlib figure.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    gdf[tc] = -gdf[tc]

    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]  # Adjust grid resolution as needed

    # Interpolate the data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')


    # Load a base map of the US
    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter out Alaska, Hawaii, and other territories to keep only the mainland US
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]

    # unary_union creates a single geometry out of MultiPolygon
    us_boundary = unary_union(us_map.geometry)
    # If the boundary is a MultiPolygon (which it will be if there are islands), we will combine them
    if isinstance(us_boundary, MultiPolygon):
        us_boundary = us_boundary.simplify(0.05)  # You might need to adjust the tolerance to your specific case
        exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms if poly.is_valid and not poly.is_empty]
    else:
        exterior_coords = [np.array(us_boundary.exterior.coords)]

    # Flatten the list of arrays into a single array and create a Path object
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    # Create the mask
    inside_us = np.array([
        us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)
    
    # Apply the mask to your grid_z
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    # Plot the base map
    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Plot the contour map with the masked grid
    if color == 'red':
        
        #range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])
        range_cmap = LinearSegmentedColormap.from_list('white_to_lightblue', ['white', '#9999FF'])
    elif color == 'green':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'green'])
    elif color == 'blue':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'blue'])

    contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    #fig.colorbar(contour, ax=ax, label='Net Benefits')

    # Set the plot limits to the bounds of the contiguous US, excluding Alaska and Hawaii
    # You may need to manually set these limits to fit the mainland US appropriately
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    
    #cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20, label=legend_label)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)  # You can adjust the fontsize as needed.

    # Customizing color bar ticks
    if legend_ticks is not None:
        cbar.set_ticks(legend_ticks)
        cbar.set_ticklabels([str(tick) for tick in legend_ticks])
    
    # Customizing plot title
    if title:
        ax.set_title(title, fontsize=22)

    # Removing frame around the map
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Customizing the axes ticks
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Removing coordinate numbers (tick labels)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # Dynamically set legend ticks to integer values within a reasonable range
    
    tick_start = gdf[tc].min()
    tick_end = gdf[tc].max()
    ticks = np.linspace(tick_start, tick_end, num=5, endpoint=True)  # Generates 5 evenly spaced ticks
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
    
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    # Show the plot
    plt.show()

def create_contour_plot_ach_reduction(gdf, data, tc, title='', legend_label='', legend_ticks=None, color = 'red'):
    """
    Create a contour plot of Air Quality Grid data with PM concentrations.

    Parameters:
    - aq_grid_gdf: GeoDataFrame with Air Quality Grid data.
    - pm_data: Dictionary with PM2.5 concentration data.
    - target_column: Column name of the value to plot.
    - contour_levels: Number of levels in the contour plot.
    - cmap_name: Name of the colormap to use.
    - linewidth: Width of the lines on the US map.
    - figsize: Size of the matplotlib figure.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    #gdf[tc] = -gdf[tc]

    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()
    # Create a regular grid to interpolate the data
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]  # Adjust grid resolution as needed

    # Interpolate the data onto the grid
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')


    # Load a base map of the US
    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    # Filter out Alaska, Hawaii, and other territories to keep only the mainland US
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]

    # unary_union creates a single geometry out of MultiPolygon
    us_boundary = unary_union(us_map.geometry)
    # If the boundary is a MultiPolygon (which it will be if there are islands), we will combine them
    if isinstance(us_boundary, MultiPolygon):
        us_boundary = us_boundary.simplify(0.05)  # You might need to adjust the tolerance to your specific case
        exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms if poly.is_valid and not poly.is_empty]
    else:
        exterior_coords = [np.array(us_boundary.exterior.coords)]

    # Flatten the list of arrays into a single array and create a Path object
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    # Create the mask
    inside_us = np.array([
        us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())
    ]).reshape(grid_x.shape)
    
    # Apply the mask to your grid_z
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    # Plot the base map
    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Plot the contour map with the masked grid
    if color == 'red':
        
        #range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])
        range_cmap = LinearSegmentedColormap.from_list('white_to_lightblue', ['white', '#9999FF'])
    elif color == 'green':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'green'])
    elif color == 'blue':
        range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['cornflowerblue', 'violet'])
        #range_cmap = LinearSegmentedColormap.from_list('white_to_red', ['white', 'violet'])

    #contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 1000), cmap=range_cmap, alpha=0.9)
    contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(50, 90, 1000), cmap=range_cmap, alpha=0.9)
    #fig.colorbar(contour, ax=ax, label='Net Benefits')

    # Set the plot limits to the bounds of the contiguous US, excluding Alaska and Hawaii
    # You may need to manually set these limits to fit the mainland US appropriately
    ax.set_xlim(-125, -66.5)
    ax.set_ylim(24.5, 49.5)
    
    #cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20, label=legend_label)
    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)  # You can adjust the fontsize as needed.

    # Customizing color bar ticks
    if legend_ticks is not None:
        cbar.set_ticks(legend_ticks)
        cbar.set_ticklabels([str(tick) for tick in legend_ticks])
    
    # Customizing plot title
    if title:
        ax.set_title(title, fontsize=22)

    # Removing frame around the map
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Customizing the axes ticks
    #ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    #ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Removing coordinate numbers (tick labels)
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    # Dynamically set legend ticks to integer values within a reasonable range
    
    tick_start = gdf[tc].min()
    tick_end = gdf[tc].max()
    ticks = np.linspace(tick_start, tick_end, num=5, endpoint=True)  # Generates 5 evenly spaced ticks
    
    ticks = [50, 60, 70, 80, 90]
    #ticks = [55, 65, 75, 85]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])
    
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    
    # Show the plot
    plt.show()

'''
def create_contour_plot(gdf, data, tc, title='', legend_label='', color='red'):
    """
    Create a contour plot of Air Quality Grid data with specified values.
    """
    gdf[tc] = gdf['GRID_KEY'].map(data)
    gdf = gdf.dropna(subset=[tc])
    gdf[tc] = (gdf[tc] * 100000).round().astype(int)

    x = [geometry.centroid.x for geometry in gdf.geometry]
    y = [geometry.centroid.y for geometry in gdf.geometry]
    z = gdf[tc].to_numpy()

    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    us_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    us_map = us_map[(us_map['name'] == 'United States of America') & (us_map['continent'] == 'North America')]
    us_boundary = unary_union(us_map.geometry)
    exterior_coords = [np.array(poly.exterior.coords) for poly in us_boundary.geoms]
    exterior_coords = np.vstack(exterior_coords)
    us_map_path = Path(exterior_coords)

    inside_us = np.array([us_map_path.contains_point((gx, gy)) for gx, gy in zip(grid_x.flatten(), grid_y.flatten())]).reshape(grid_x.shape)
    grid_z_masked = np.ma.masked_where(~inside_us, grid_z)

    fig, ax = plt.subplots(figsize=(10, 15))
    us_map.plot(ax=ax, color='white', edgecolor='black', linewidth=1.5)

    # Define color map based on the input color
    color_map = {'red': 'red', 'green': 'green', 'blue': 'blue'}
    range_cmap = LinearSegmentedColormap.from_list('custom', ['white', color_map.get(color, 'red')])

    contour = ax.contourf(grid_x, grid_y, grid_z_masked, levels=np.linspace(z.min(), z.max(), 100), cmap=range_cmap, alpha=0.9)

    cbar = fig.colorbar(contour, ax=ax, shrink=0.4, aspect=20)
    cbar.set_label(legend_label, fontsize=15)

    # Dynamically set legend ticks to integer values within a reasonable range
    tick_start = int(np.floor(z.min() / 1000)) * 1000
    tick_end = int(np.ceil(z.max() / 1000)) * 1000
    ticks = np.linspace(tick_start, tick_end, num=5, endpoint=True)  # Generates 5 evenly spaced ticks
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])

    if title:
        ax.set_title(title, fontsize=22)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    plt.show()
'''

#%%

def plot_histogram(data, title, xlabel, ylabel, color='skyblue', edgecolor='black', save = False):
    """
    Plots a histogram with the given data and customization options.

    Parameters:
    - data: Pandas Series containing the data to plot.
    - title: String, title of the histogram.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    - color: String, color of the histogram bars. Default is 'skyblue'.
    - edgecolor: String, color of the bar edges. Default is 'black'.
    """
    plt.figure(figsize=(8, 6))  # Set figure size for better visibility
    
    if isinstance(data, list):
        data = pd.Series(data)
    data.hist(color=color, edgecolor=edgecolor)
    
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Optional: Add grid for better readability
    plt.grid(axis='y', alpha=0.75)

    plt.show()
    
def plot_boxplot(data, title, xlabel, ylabel, color='skyblue', save=False, filename='boxplot.png'):
    """
    Plots a box plot with the given data and customization options.

    Parameters:
    - data: List or Pandas Series containing the data to plot.
    - title: String, title of the box plot.
    - xlabel: String, label for the x-axis.
    - ylabel: String, label for the y-axis.
    - color: String, color of the box plot elements. Default is 'skyblue'.
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(8, 6))  # Set figure size for better visibility
    
    # Creating the box plot
    box = plt.boxplot(data, patch_artist=True)  # 'patch_artist' must be True to fill with color
    plt.setp(box['boxes'], color=color, facecolor=color)  # Setting color of the boxes
    plt.setp(box['whiskers'], color=color)  # Setting color of the whiskers
    plt.setp(box['caps'], color=color)  # Setting color of the caps
    plt.setp(box['medians'], color='black')  # Setting color of the medians
    
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Saving the plot if required
    if save:
        plt.savefig(filename)

    plt.show()
    
def plot_boxplots(data, title, xlabels, ylabel, colors='skyblue', save=False, filename='boxplot.png'):
    """
    Plots side-by-side box plots for given datasets on a single plot.

    Parameters:
    - data: List of lists or Pandas Series to plot side by side.
    - title: String, title of the box plot.
    - xlabels: List of strings, labels for each box plot on the x-axis.
    - ylabel: String, label for the y-axis.
    - colors: List of strings or a single string for the color of the box plot elements. Default is 'skyblue'.
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(8, 6))  # Set figure size for better visibility

    # Check if colors is a single color and expand it to the length of data if needed
    if isinstance(colors, str):
        colors = [colors] * len(data)

    # Creating the box plot
    box = plt.boxplot(data, patch_artist=True, positions=range(1, len(data) + 1), showfliers = False)

    # Set colors and properties for each box
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.setp(box['whiskers'], color='black')
    plt.setp(box['caps'], color='black')
    plt.setp(box['medians'], color='black')

    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.ylabel(ylabel, fontsize=12)
    
    # Setting x-axis labels
    plt.xticks(range(1, len(data) + 1), xlabels, fontsize=10)
    plt.ylim(bottom=0)
    # Saving the plot if required
    if save:
        plt.savefig(filename)

    plt.show()
    
def plot_grouped_boxplots(data, title, group_labels, subplot_labels, ylabel, colors=('skyblue', 'blue', 'green','gray', 'red'), save=False, filename='boxplot.png'):
    """
    Plots grouped box plots with subgroups for given datasets on a single plot.

    Parameters:
    - data: List of lists, where each list contains multiple datasets for subgroups.
    - title: String, title of the box plot.
    - group_labels: List of strings, labels for each main group on the x-axis.
    - subplot_labels: List of strings, labels for each subgroup within the main groups.
    - ylabel: String, label for the y-axis.
    - colors: Tuple of strings, colors for each subgroup. Default is ('skyblue', 'green').
    - save: Boolean, if True, saves the plot to a file. Default is False.
    - filename: String, filename to save the plot. Default is 'boxplot.png'.
    """
    plt.figure(figsize=(10, 6))  # Set figure size for better visibility
    num_groups = len(data)
    num_subgroups = len(data[0])
    width = 0.35  # width of each boxplot within a group
    
    # Create positions for each subgroup
    positions = []
    for i in range(num_groups):
        start = i * num_subgroups * 3
        positions += [start + j * width * 3 for j in range(num_subgroups)]
    
    # Flatten the data and adjust colors accordingly
    flat_data = [item for sublist in data for item in sublist]
    colors = list(colors) * num_subgroups

    # Create box plots
    box = plt.boxplot(flat_data, patch_artist=True, positions=positions, widths=width, showfliers = False)

    # Set colors for each subgroup
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Set properties for other elements
    plt.setp(box['whiskers'], color='black')
    plt.setp(box['caps'], color='black')
    plt.setp(box['medians'], color='black')

    # Add titles and labels
    plt.title(title, fontsize=16)
    plt.ylabel(ylabel, fontsize=14)
    
    # Set custom x-axis labels
    ax = plt.gca()
    tick_positions = [np.mean(positions[i:i+num_subgroups]) for i in range(0, len(positions), num_subgroups)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(group_labels)

    # Add legend for subgroups
    plt.legend([ box["boxes"][i] for i in range(0, num_subgroups)], subplot_labels, loc='upper right')

    # Save the plot if required
    if save:
        plt.savefig(filename)

    plt.show()