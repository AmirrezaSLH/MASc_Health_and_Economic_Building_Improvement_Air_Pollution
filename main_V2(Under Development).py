# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 14:01:13 2023

@author: asalehi
"""
import pandas as pd
import geopandas as gpd

def init_aq_grids(file_path: str = 'Data/AQgrid.gpkg') -> gpd.GeoDataFrame:
    """
    Load the AQ Grids from a GeoPackage file.

    :param file_path: Path to the AQ Grids GeoPackage file.
    :return: A GeoDataFrame containing AQ Grids.
    """
    AQ_Grid_gdf = gpd.read_file(dir)
    return AQ_Grid_gdf

def init_county(file_path: str = 'Data/County_Main_Land.gpkg') -> gpd.GeoDataFrame:
    """
    Load the Counties from a GeoPackage file.

    :param file_path: Path to the Counties GeoPackage file.
    :return: A GeoDataFrame containing County data.
    """
    County_gdf = gpd.read_file(dir)
    return County_gdf

def init_pm_concentrations(file_path: str = 'Data/PM_Grid_Daily_Concentration_1981_2010.csv') -> pd.DataFrame:
    """
    Load PM2.5 Concentration per Grid Cell per year from a CSV file.
    The 'Values' column contains a string of 365 daily concentrations.

    :param file_path: Path to the PM Concentrations CSV file.
    :return: A DataFrame containing PM2.5 concentrations.
    """
    
    PM_Concentrations = pd.read_csv(dir)
    return PM_Concentrations

def init_buildings_stock(file_path: str = 'Data/Buildings_Stock.csv') -> pd.DataFrame:
    """
    Load the Buildings Stock from a CSV file.

    :param file_path: Path to the Buildings Stock CSV file.
    :return: A DataFrame containing Buildings Stock data.
    """
    
    Buildings_Stock_df = pd.read_csv(file_path)
    return Buildings_Stock_df

def init_population(file_path: str = 'Data/Population.csv') -> pd.DataFrame:
    """
    Load Population data from a CSV file.
    The current state of the model only uses 2000 population data based on Leuple.

    :param file_path: Path to the Population CSV file.
    :return: A DataFrame containing Population data.
    """
    
    Population_df = pd.read_csv(file_path)
    return Population_df