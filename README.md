

This repository contains the **original spatial health and air quality modeling framework**, designed to analyze the impact of building airtightness interventions on indoor air quality (IAQ), health outcomes (e.g., mortality reduction), and associated cost-benefit metrics across the U.S.

📅 **Last updated:** August 2024  
🔁 **Project continued at:** [Health_IAQ_Airtightness_Framework](https://github.com/AmirrezaSLH/Health_IAQ_Airtightness_Framework)

---

## 🗂️ Project Structure

### `main.py`
The main script that ties together the core analysis. It:
- Initializes data sources (population, AQ grids, leakage/ACH relationships)
- Defines key physical and statistical functions (e.g., `ACH50_to_Finf`)
- Implements the Monte Carlo simulation via `run_MCS()`
- Calculates national averages for PM2.5 reductions and benefits

---

### `County_Grid_Percentage_Mapping.py`
Maps county-level data to spatial air quality (AQ) grid cells using geographic overlap:
- Loads grid and county geometries
- Computes percent overlap of counties with AQ grid cells
- Outputs mappings as JSON files for use in downstream analysis

---

### `Analysis_Plots.py`
Contains functions for generating national and spatial plots including:
- Histograms and boxplots of mortality, costs, and benefits
- Contour maps of PM2.5, net benefits, and incidence reductions
- Visualizations are based on gridded GeoDataFrames (`geopandas`)

---

### `Scenario Analysis.py`
Evaluates various retrofit or intervention scenarios (e.g., 20%, 40%, 60% airtightness reductions):
- Defines intervention logic (e.g., `reduction_20`)
- Simulates spatial and national health + cost outcomes
- Uses `generate_maps` and `generate_box_plots` to visualize impact
- Aggregates spatial data using population weighting

---

### `Uncertainty_Analysis.py`
Performs deep Monte Carlo simulations (e.g., 5,000 iterations) to analyze uncertainty:
- Focuses on robust statistics: 2.5–97.5 percentiles, mean, min/max
- Repeats the same scenario many times for uncertainty bounds
- Outputs distributions of national health and cost indicators

---

## 📊 Data Folder Structure

The project relies on several datasets, including:
- `AQgrid.gpkg`: Air Quality grid shapefile
- `County_Main_Land.gpkg`: U.S. mainland counties shapefile
- `Buildings_Stock.csv`: Detailed building data including ACH50, floor area, type, and year
- Multiple JSONs:
  - `County_to_Grid_percentage_mapping.json`
  - `ACH50_Grid_Vintage_BT.json`
  - `population_buildingtype_grid_mapping.json`

These are used to assign building data spatially, calculate weighted averages, and visualize results geographically.

---

## 📌 Note

This repository is an **initial version** and has been **continued and further developed here**:

🔗 [Health_IAQ_Airtightness_Framework](https://github.com/AmirrezaSLH/Health_IAQ_Airtightness_Framework)

Please refer to the new repository for updated data structures, features, and visualizations.

---

## 🧑‍💻 Author

Developed by **Amirreza Salehi** at the University of Waterloo, Saari Lab.

---

## 📄 License

MIT License
