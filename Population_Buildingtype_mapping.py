# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 14:48:06 2024

@author: asalehi
"""

import pandas as pd
import json
import geopandas as gpd

directory = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Data\Population_Building\Occupant per building type 2010 county\ACSDT5Y2010.B25033-Data.csv'

#%%
data = pd.read_csv(directory)
data.columns
data = data.drop(data.iloc[0].name)

columns_to_drop = ['NAME', 'B25033_001M', 'B25033_002M', 'B25033_003M' , 'B25033_004M', 'B25033_005M', 'B25033_006M'
                   , 'B25033_007M', 'B25033_008M', 'B25033_009M', 'B25033_010M', 'B25033_011M', 'B25033_012M', 'B25033_013M'
                   ,'Unnamed: 28']

rename = { 'B25033_001E' : 'TOTAL',
           'B25033_002E' : 'TOTAL_O',
           'B25033_003E' : 'SDA_O',
           'B25033_004E' : '2-4_O',
           'B25033_005E' : '5P_O',
           'B25033_006E' : 'M_O',
           'B25033_007E' : 'OTHER_O',
           'B25033_008E' : 'TOTAL_R',
           'B25033_009E' : 'SDA_R',
           'B25033_010E' : '2-4_R',
           'B25033_011E' : '5P_R',
           'B25033_012E' : 'M_R',
           'B25033_013E' : 'OTHER_R'}

data = data.drop( columns = columns_to_drop )
data = data.rename(columns=rename)

county_list = data['GEO_ID'].to_list()
print(county_list)

population_buildingtype_county_mapping = {}
for idx, row in data.iterrows():
    c = row['GEO_ID']
    c = c.replace('0500000US', 'F')
    population_buildingtype_county_mapping[c] = {}
    Total =int( row['TOTAL'])
    SDA = int(row['SDA_O']) + int(row['SDA_R'])
    U2_4 = int(row['2-4_O']) + int(row['2-4_R'])
    P5 = int(row['5P_O']) + int(row['5P_R'])
    M = int(row['M_O']) + int(row['M_R'])
    Other = int(row['OTHER_O']) + int(row['OTHER_R'])
    
    '''
    population_buildingtype_county_mapping[c]['SDA'] = (SDA / Total) * 100
    population_buildingtype_county_mapping[c]['2-4'] = (U2_4 / Total) * 100
    population_buildingtype_county_mapping[c]['5P'] = (P5 / Total) * 100
    population_buildingtype_county_mapping[c]['M'] = (M / Total) * 100
    population_buildingtype_county_mapping[c]['OTHER'] = (Other / Total) * 100
    '''
    
    population_buildingtype_county_mapping[c]['TOTAL'] = Total
    population_buildingtype_county_mapping[c]['SDA'] = SDA
    population_buildingtype_county_mapping[c]['2-4'] = U2_4
    population_buildingtype_county_mapping[c]['5P'] = P5
    population_buildingtype_county_mapping[c]['M'] = M
    population_buildingtype_county_mapping[c]['OTHER'] = Other
 
out_dir = r'C:\Users\asalehi\OneDrive - University of Waterloo\Documents - SaariLab\CVC\Buildings\Amirreza\Adaptation\Github_Adaptation\MASc_Waterloo_Adaptation_Buildings\Data\\'
filename = 'population_buildingtype_county_mapping.json'

with open(out_dir + filename, 'w') as file:
    json.dump(population_buildingtype_county_mapping, file, indent=4)
    
    
#%%

def init_AQ_Grids( dir = 'Data/AQgrid.gpkg'):
    #This Function Loads the AQ Grids
    AQ_Grid_gdf = gpd.read_file(dir)
    AQ_Grid_gdf['GRID_KEY'] = [(col, row) for col, row in zip(AQ_Grid_gdf['COL'], AQ_Grid_gdf['ROW'])]

    return AQ_Grid_gdf

def init_Country_to_Grid_Percentage_Mapping(  dir = 'Data/County_to_Grid_percentage_mapping.json' ):
    with open(dir, 'r') as file:
        Country_to_Grid_Percentage_Mapping = json.load(file)
    
    return Country_to_Grid_Percentage_Mapping

Country_to_Grid_Percentage_Mapping = init_Country_to_Grid_Percentage_Mapping()

with open(out_dir + filename, 'r') as file:
    population_buildingtype_county_mapping = json.load(file)

population_buildingtype_grid_mapping = {}
i = 0
for c in population_buildingtype_county_mapping.keys():
    if c in Country_to_Grid_Percentage_Mapping.keys():
        Grid_Percentage_Mapping = Country_to_Grid_Percentage_Mapping[c]
    else:
        i+= 1
        continue
    
    for g in Grid_Percentage_Mapping.keys(): 
        if g in population_buildingtype_grid_mapping.keys():
            #a=0
            for bt in population_buildingtype_county_mapping[c].keys():
                population_buildingtype_grid_mapping[g][bt] += ( population_buildingtype_county_mapping[c][bt] * Grid_Percentage_Mapping[g])
        else:
            population_buildingtype_grid_mapping[g] = {}
            for bt in population_buildingtype_county_mapping[c].keys():
                population_buildingtype_grid_mapping[g][bt] = (population_buildingtype_county_mapping[c][bt] * Grid_Percentage_Mapping[g])

filename_2 = 'population_buildingtype_grid_mapping.json'

with open(out_dir + filename_2, 'w') as file:
    json.dump(population_buildingtype_grid_mapping, file, indent=4)
    
population_percentage_buildingtype_grid_mapping = {}
for g in population_buildingtype_grid_mapping.keys():
    population_buildingtype_mapping = population_buildingtype_grid_mapping[g]
    population_percentage_buildingtype_grid_mapping[g] = {}
    
    for bt in population_buildingtype_mapping.keys():
        if bt == 'TOTAL':
            continue
        else:
            population_percentage_buildingtype_grid_mapping[g][bt] = population_buildingtype_mapping[bt]/population_buildingtype_mapping['TOTAL']

filename_3 = 'population_buildingtype_grid_percentage_mapping.json'

with open(out_dir + filename_3, 'w') as file:
    json.dump(population_percentage_buildingtype_grid_mapping, file, indent=4)
   
#just testing
i = 0          
for g in population_buildingtype_grid_mapping.keys():
    i += population_buildingtype_grid_mapping[g]['TOTAL']
    
print(i)

#just testing

i = 0          
for c in population_buildingtype_county_mapping.keys():
    i += population_buildingtype_county_mapping[c]['TOTAL']
    
print(i)