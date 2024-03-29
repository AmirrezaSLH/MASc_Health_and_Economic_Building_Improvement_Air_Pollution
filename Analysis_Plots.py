# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:01:05 2024

@author: asalehi
"""
#%%
import numpy as np
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

#%%

import main_25_March

def ACH50_to_Finf(ach50, P=0.97, K=0.39, F=20):
    """
    Convert blower door test results (ACH_50) to natural infiltration rate (Finf).

    This function estimates the natural infiltration rate based on the air change rate at
    50 Pascals (ACH_50) from blower door tests. It accounts for the penetration factor of
    particulate matter (P), the deposition rate (K), and the conversion factor (F) to
    adjust ACH_50 to ACH_natural.

    Parameters:
    - ACH_50 (float): Air Change per Hour at 50 Pascals from a blower door test.
    - P (float): Penetration factor representing how much particulate matter gets inside. Default is 0.97.
    - K (float): Deposition factor representing how fast particulate matter settles. Default is 0.39.
    - F (float): Conversion factor to relate ACH_50 to natural air change rate. Default is 20.

    Returns:
    - float: The natural infiltration rate (Finf).

    Notes:
    - The user of this function should consider local building codes and literature for appropriate values of P, K, and F.
    - The default values provided may not be suitable for all building types and climates.
    """
    # Convert ACH_50 to natural air change rate (ACH_natural)
    ach_natural = ach50 / F
    
    # Calculate natural infiltration rate (Finf)
    Finf = (P * ach_natural) / (ach_natural + K)
    
    return Finf

def finf( ach50_segment_dict ):
    FINF_baseline = {key: ACH50_to_Finf(0.8 * value) for key, value in ach50_segment_dict.items()}
    return FINF_baseline

a1, b1, c1, delta_risk = main_25_March.run_model( intervention_function = finf)
#%%
#Setting the name of save folder
from datetime import datetime
# Get the current date
current_date = datetime.now()
# Format the date as 'MonthDay' without spaces
date_str = current_date.strftime("%B%d")
image_base_save_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Plots'
image_save_dir = os.path.join(image_base_save_dir, date_str)

def plot_geodataframe(gdf, title, column_to_plot, cmap='jet', figsize=(10, 6),
                      title_fontsize=16, legend_fontsize=12, legend_tag = 'dollar', save = False):
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Plot the GeoDataFrame
    gdf.plot(column=column_to_plot, ax=ax, legend=True, cmap=cmap,
             legend_kwds={'label': legend_tag})

    # Remove the axis for a cleaner look
    ax.set_axis_off()
    
    ax.set_title(title, fontsize=title_fontsize)
    
    # Adjust the layout to make space for the title and legend
    plt.subplots_adjust(top=0.85, bottom=0.2)
    if save == True:
        plt.savefig(image_save_dir + '/'+ title, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()

def plot_geodataframe_large(gdf, title, column_to_plot, cmap='jet', figsize=(10, 6),
                      title_fontsize=16, legend_fontsize=12, legend_tag = 'dollar', save = False):
    """
    Plots a GeoDataFrame with enhancements, including a title and a legend in millions.
    
    Parameters:
    - gdf: GeoDataFrame to plot.
    - title: Title for the plot.
    - column_to_plot: The name of the column in the GeoDataFrame to base the plot on.
    - cmap: Colormap for the plot.
    - figsize: Size of the figure.
    - title_fontsize: Font size for the plot title.
    - legend_fontsize: Font size for the legend.
    """
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot the GeoDataFrame
    plot = gdf.plot(column=column_to_plot, ax=ax, legend=True, cmap=cmap,
                    legend_kwds={'label': 'Milion'+legend_tag, 'format': mticker.FuncFormatter(lambda v, pos: f'{v * 1e-6:,.0f}  ')})
    
    # Add the title to the plot
    ax.set_title(title, fontsize=title_fontsize)
    
    # Remove the axis for a cleaner look
    ax.set_axis_off()
    
    # Customize the legend
    # Get the figure's colorbar instance and modify its label
   # fig.colorbar(plot.get_children()[0], ax=ax, 
   #              format=mticker.FuncFormatter(lambda v, pos: f'{v * 1e-9:,.0f} M'))
    
    # Adjust the layout to make space for the title and legend
    
    plt.subplots_adjust(top=0.85, bottom=0.2)
    if save == True:
        plt.savefig(image_save_dir + '/'+ title, dpi=300, bbox_inches='tight')
    # Show the plot
    plt.show()
    
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
    data.hist(color=color, edgecolor=edgecolor)
    
    # Add titles and labels
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    
    # Optional: Add grid for better readability
    plt.grid(axis='y', alpha=0.75)

    plt.show()
