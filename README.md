# resite_ip

This branch of the `resite_ip` repository contains Python scripts used to deploy RES assets based on the different siting strategies
investigated in Chapter 9 ("On the role of complementarity in siting renewable power generation assets and its economic 
implications for power systems") of the following publication: "Complementarity of Variable Renewable Energy Sources", 
ISBN: 9780323855273, available [here](https://www.elsevier.com/books/complementarity-of-variable-renewable-energy-sources/jurasz/978-0-323-85527-3).
## Setup

This module runs in any `Python` version older than 3.6. In addition, all siting scripts are written in `Julia`, so a `Julia` 
version no older than 1.5 is also required. Finally, the tool requires to installation of the `gurobi` solver (with a valid license).
Once `Python`, `Julia` and `gurobi` are installed on the machine and assuming that the code was directly obtained
via the provided tag ling, the installation process requires the following steps:

* create a conda environment from the `requirements.yml` file 
* install of `Julia`-specific modules via the `import Pkg; Pkg.add()` command
* download input [data](https://dox.uliege.be/index.php/s/L9jH5aCQdZ7ie4W) folder and set-up data paths in the `config_model.yaml` file
   
## Repository structure

This structure of this package is as follows:
* `resite_ip` as a container for:
  * the `src` folder, which includes 
    * the `.py` files taking care of pre- and post-processing
    * the `jl` folder containing the `.jl` routines defining the siting heuristics
  * the `config` files
  * auxiliary files required to set up the package
  
The `data` folder (whose path must be specified in the first line of `config_model.yml`) is provided in a different zenodo-like repository. 

## Typical run
Once all dependencies are installed, in order to run the model, the following steps must be taken:
* Configure run via the `config` files
* Run `python main.py` from the `src` folder

Results of these runs are stored in an `output` directory at the same level as the `data` folder.

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).


