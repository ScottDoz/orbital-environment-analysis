# orbital-environment-analysis
Analysis of the Earth orbital environment, and the objects within it

## About
* Methods to import satellite TLE data
* Computation of orbital parameters
* Clustering of objects
* Density estimation
* Pair-wise distance metrics between objects
* Visualizations
* Detectibility, Identification, Trackability (DIT) analysis

## Installation
Some code in this repo requires the use of the NASA GMAT Python API. This API currently only supports Python 3.7. Suggest creating a new Python 3.7 virtual environment, and installing the requirements.txt file.

### Data Directory Setup
Satellite TLE data are saved to the user's local disk at ~/satellite_data/Data. This directory will be automatically created on first call to any of the satellite data loading methods. The user can set an alternative location by creating an envirnment variable 'SATELLITE_DATA'.

The SatelliteData module provides methods to load experimental data from Nair, Vishnu, 2021, "Statistical Families of the Trackable Earth-Orbiting Anthropogenic Space Object Population in Their Specific Orbital Angular Moment Space", https://doi.org/10.18738/T8/LHX5KM, Texas Data Repository, V2.

To use this data, visit https://doi.org/10.18738/T8/LHX5KM. Download the data and unzip the json files into ~satellite_data/Data/dataverse_files (alternatively DATA_DIR/dataverse_files if you have set your own data directory).

### Create config.ini
Satellite TLE data is downloaded from Spacetrack.com via the spacetrack Python API. Access to data requires registering an email and password at www.space-track.org.

Once registered, create a new file 'config.ini' in the main directory (../orbital-environment-analysis/src/OrbitalAnalysis) and populate with the following structure, replaceing your email and password.
```
[Spacetrack]
email = user@email.com
pw = password
```
