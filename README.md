# resite_ip

This repository hosts the main development work around `resite_ip`, a tool relying on integer programming to site
renewable generation assets based on various deplyoment criteria.

## Setup

The code runs in `Python 3.6` or older. In addition, running the code requires installation of `Julia 1.5` or older.
Furthermore, the some of the siting algorithm available in this repository will require a branch-and-bound solver. Currently,
only `gurobi` (with a valid license) is supported.

## Repository structure

This structure of this package is as follows:
* `resite_ip` as a container for:
  * the `src` folder, which includes 
    * the `.py` files taking care of pre- and post-processing
    * the `jl` folder containing the `.jl` routines defining the siting heuristics
  * the `config` files
  * auxiliary files required to set up the package
  
The `data` folder (whose path must be specified in the first line of `config_model.yml`) is needed to instantiate the 
model. Currently, the data available covers solely the European continent and is available [here](https://dox.uliege.be/index.php/s/L9jH5aCQdZ7ie4W). 
In order to run the model and assuming that this repository has been obtained via `git clone` the following steps must be taken:

* Install all package dependencies via `conda env create -f requirements.yml`
* Configure run via the `config` files
* Run `python main.py` from the `src` folder

Results of these runs are stored in an `output` directory in the `data` folder.

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).


