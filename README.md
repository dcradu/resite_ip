# resite_ip

This branch of the `resite_ip` repository contains Python scripts used to deploy RES assets based on the different siting strategies
investiaged in Chapter 9 ("On the role of complementarity in siting renewable power generation assets and its economic 
implications for power systems") of the following publication: "Complementarity of Variable Renewable Energy Sources", 
ISBN: 9780323855273, available [here](https://www.elsevier.com/books/complementarity-of-variable-renewable-energy-sources/jurasz/978-0-323-85527-3).
## Setup

This module runs in any 3.x version of Python. The Python packages required to run this module are listed 
in the `requirements.yml` file included in the repository. Their installation can be easily achieved via the the following
command that builds a separate environment for this module from the `yml` file:
   
    conda env create -f environment.yml

The tool requires to installation of the `gurobi` solver.
   
## Minimal working example

This structure of this package is as follows:
* `resite_ip` as a container for:
  * the `src` folder, which includes 
    * the `.py` files taking care of pre- and post-processing
    * the `jl` folder containing the `.jl` routines defining the siting heuristics
  * the `config` files
  * auxiliary files required to set up the package
  
The `data` folder (whose path must be specified in the first line of `config_model.yml`) is provided in a different zenodo-like repository. In order to run the model, the following steps must be taken:
* Ensure input data is available according to the structure provided in the `data` folder
* Configure run via the `config` files
* Run `python main.py` from the `src` folder

Results of these runs are stored in an `output` directory at the same level as the `data` folder.

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).


