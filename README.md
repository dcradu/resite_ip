# resite_ip

This branch of the `resite_ip` repository contains Python scripts used to pre- and post-process application-related data
for the max k-multicover problem introduced in the following publication: "Siting Renewable Power Generation Assets with Combinatorial Optimisation", Mathias Berger et al., Optimization Letters, 2021. DOI: https://doi.org/10.1007/s11590-021-01795-0

## Setup

This module runs in any 3.x version of Python. The Python packages required to run this module are listed 
in the `requirements.yml` file included in the repository. Their installation can be easily achieved via the the following
command that builds a separate environment for this module from the `yml` file:
   
    conda env create -f environment.yml
    
In addition, this repository requires the cloning of the `maxk_multicover` repository containing a set of `Julia` scripts and 
available [here](https://gitlab.uliege.be/smart_grids/public/maxk_multicover).
The latter repository should be cloned in the same root folder where the `resite_ip` repository is used. Once this is done,
the `maxk_multicover` folder has to be added to the PYTHONPATH.

The tool also requires to installation of the `gurobi` solver. The current version of the repository is tested and the outcomes of different
algorithms are benchmarked against Gurobi 9.1.
   
## Repository content

The repository contains, besides three `yaml` files used for model configuration and environment installation, the following four scripts within the `src` folder:
1. The `era5data.py` script provides the code to download reanalysis data via the cdsAPI of ECMWF 
2. The `helpers.py` script contains various methods for data manipulation and pre-processing
3. The `tools.py` script gathers the main functions enabling the conversion of raw reanalysis data in a structured form, ready to be fed to the various siting routines
4. The `main.py` is mainly used to call the different siting routines embedded in this repository

## License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).


