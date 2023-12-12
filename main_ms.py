# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:47:35 2023

@author: asalehi
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

Out_Put_Dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Github_Adaptation\MASc_Waterloo_Adaptation_Buildings\Data'


def init_AQ_Grids( dir = 'Data/AQgrid.gpkg'):
    #This Function Loads the AQ Grids
    AQ_Grid_gdf = gpd.read_file(dir)
    return AQ_Grid_gdf

def init_County( dir = 'Data/County_Main_Land.gpkg'):
    #This Function Loads the Counties
    County_gdf = gpd.read_file(dir)
    return County_gdf

def init_PM_Concentrations( dir = 'Data/PM_Grid_Daily_Concentration_1981_2010.csv'):
    # This Function loads PM2.5 Concentration per Grid Cell per year. 
    #The Values column contains a string of 365 daily concentration
    PM_Concentrations = pd.read_csv(dir)
    return PM_Concentrations

def init_Buildings_Stock( dir = 'Data/Buildings_Stock.csv'):
    #This Function Loads the Buildings Stock
    Buildings_Stock_df = pd.read_csv(dir)
    return Buildings_Stock_df



def init_Population():
    # The reference population depends on what we are studying. 
    # If using Lepeule to calculate mortalities:
    # "C:\Users\matts\OneDrive - University of Waterloo\Sparks\PNAS\WorkingFolder\Data\General\pop_lepeule.csv"
    # If using Krewski to calculate mortalities:
    # "C:\Users\matts\OneDrive - University of Waterloo\Sparks\PNAS\WorkingFolder\Data\General\pop_krewski.csv"
    return

def PM_to_Grid_gdf(PM_Concentrations = init_PM_Concentrations() , AQ_Grid_gdf = init_AQ_Grids()):
    # PM Concentrations should be assigned to grid cells already?
    # If you are assigning them to the AQ_Grid_gdf, then the .loc method should work:
    # Iterate over all rows r and columns c:
    # AQ_Grid_gdf.loc[(AQ_Grid_gdf['ROW] == r) & (AQ_Grid_gdf['COL'] == c), 'PM'] =\
    # PM_Concentrations.loc[(PM_Concentrations['ROW'] == r) & (PM_Concentrations['COL'] == c), 'PM'].item()
    
    return

def ACH50_to_County(Buildings_Stock_df, County_gdf = init_County() ):
    #This Function assigns average value of ACH50 to each County
    #The Current Method for Averaging is per building
    County_List = County_gdf['FIPS'].to_list()
    Buildings_Stock_df_Copy = Buildings_Stock_df.copy(deep = True)
    ACH50_dict = dict()
    for C in County_List:
        Target_Stock = Buildings_Stock_df_Copy[ Buildings_Stock_df_Copy['FIPS'] == C]
        ACH50_mean = Target_Stock['ACH50'].mean()
        ACH50_dict.update({ C : ACH50_mean})
        
    County_gdf['ACH50_mean'] = County_gdf['FIPS'].map(ACH50_dict)
    return County_gdf

def ACH50_to_INF():
    # There are a few ways to do this. One is shown below, from Amy Li
    # ACH_natural = ACH50 / 20
    # P = 0.97 # Penetration factor (how much PM gets inside)
    # k = 0.39 # Deposition factor (how fast PM settles)
    # FINF = (P * ACH_natural) / (ACH_natural + K)
    # MS note, I think the P value is high here and gives high FINF values
    return


def County_to_Grid():
    # I've done something similar in 
    # "C:\Users\matts\OneDrive - University of Waterloo\Sparks\PNAS\WorkingFolder\Code\General\counties_to_grid_Updated.py"
    return

def Delta_Exposure():
    # Happy to chat about this if you want.
    return

def Delta_Risk():
    # I think the equation in my proposal might be applicable?
    # Will be larger reductions in exposure, so potentially not linear delta risk?
    # Need to check the literature on that.
    return


def PV_Convertor(Interest_Rate, Base_Year, Secondary_Year, Value):
    
    #Interest Rate in percentage : Interest_Rate = 10
    
    Delta_Year = Secondary_Year - Base_Year
    PV = Value / (1 + (Interest_Rate/100)) ** Delta_Year
    return PV

def Costs_Retrofit():
    return

def Benefit_Retrofit():
    return

def NPV_Calculation( PV_Cost, PV_Benefit):
    NPV = PV_Benefit - PV_Cost
    return NPV

def GRID_NPV():
    return

def iterate_GRID():
    return

#%% Main

iterate_GRID

#%% Tests
test = PV_Convertor(10, 2000, 2001, 110)
print(test)

test2 = init_AQ_Grids()
print(test2.crs)
test2.plot()

test3 = init_County()
print(test3.crs)
test3.plot()

test4 = init_Buildings_Stock()
test4.head()

test5 = ACH50_to_County(test4, test3)
test6 = ACH50_to_County(test4, test3)

# Assuming test6 is your GeoDataFrame
ax = test6.plot(column='ACH50_mean', legend=True)
# Set the title of the plot
ax.set_title("ACH50")
# Show the plot
plt.show()

test7 = init_PM_Concentrations()
