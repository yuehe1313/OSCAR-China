# OSCAR-China Model

## Overview
The **OSCAR-China model** is a specialized regional adaptation of the global **OSCAR v3.2 model**, developed specifically for application within China. This version operates with a spatial resolution of **0.5°×0.5°** and is tailored to assess the impacts of land use and land-cover changes (LULCC) on the carbon cycle. Designed as an "offline" model, it focuses solely on the land carbon cycle without coupling to other components of the Earth system. The model incorporates historical LULCC from 1900 to 2018 and scenario projections from 2019 to 2100, reflecting China’s national forestation policies. For access to the core OSCAR v3.2 code, visit [OSCAR v3.2 on GitHub](https://github.com/tgasser/OSCAR).

## Data Dimensions
OSCAR-China utilizes several key dimensions for organizing its input and output data:
- **year**: Represents the temporal resolution along the time axis.
- **config**: Accounts for various Monte Carlo simulation elements, facilitating uncertainty analysis with N=1,000 simulations.
- **reg_land**: Merges latitude and longitude into 0.5° pixels for efficient data processing and analysis. During the preprocessing of the input datasets, the latitude and longitude dimensions are stacked to create the reg_land dimension. After model execution, outputs can be reverted to the original latitude and longitude dimensions using the xr_reshape function.
- **bio_land**: Classifies land carbon-cycle biomes, including forest, non-forest, cropland, pasture, and urban.
- **bio_from**: Represents the original biomes of the land-use transitions.
- **bio_to**: Represents the destination biomes of the land-use transitions.
- **box_hwp**: Tracks pools of harvested wood products.

## Drivers
The model requires specific forcing data to run:
- **D_CO2**: Global atmospheric CO2 concentrations, offset by the preindustrial value used in OSCAR-China (in ppm).
- **D_Tl**: Observation-based air temperature, offset by the average over the 1901-1920 period, assumed to reflect a preindustrial condition (in °C).
- **D_Pl**: Observation-based precipitation, offset by the average over the 1901-1920 period, assumed to reflect a preindustrial condition (in mm).
- **d_Acover**: Net land cover area transitions from one biome to another during the annual time step (in Mha yr-1).
- **d_Hwood**: Biomass harvested from woody biomes (in Pg C yr-1).

## Data Files and Accessibility
The OSCAR-China model’s GitHub repository exclusively contains the model scripts. All input and output data, including historical data (1900-2018) and future predictions (2019-2100), are hosted in netCDF (*.nc) format on [Zenodo](**URL to Zenodo repository**). These output data files represent constrained means and standard deviations from 1,000 configurations, refined using empirical constraints with observation-based forest vegetation carbon stock data from the 9th National Forest Inventory and net land carbon sink data from the RECCAP2 initiative for China.
