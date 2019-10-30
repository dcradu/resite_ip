# resite

This file provides the information required to set up the __resite__ tool. Also, it provides a minimal example for siting
RES assets.  

# Setup

This module runs in any 3.x version of Python. The Python packages required to run this module are listed 
in the `requirements.yml` file included in the repository. Their installation can be easily achieved via the the following
command that builds a separate environment for this module from the `yml` file:
   
    conda env create -f environment.yml
    
In addition, the tool also requires the installation of the `pypsa` module (details [here](https://pypsa.org/)) for model
development purposes which are detailed in the `models.py` script. Since one of the latest versions of this module is 
required, its installation has to be carried out manually via

	pip install git+https://github.com/PyPSA/PyPSA.git

Lastly, the tool requires to installation of an MILP solver. At the time of writing, `gurobi` is the only solver this 
model supports (see installation details [here](https://www.gurobi.com/documentation/8.1/remoteservices/installation.html).

At the time of writing, the model readily supports (given the availability of already downloaded data
[here](https://dox.uliege.be/index.php/s/trys2xY7j9JsQ3z) and the set-up within `tools.py`) siting assessments in 
Europe, North Africa, Middle East, Greenland and the USA.

   
# Example run

The module structure includes five folders. The `input data` folder contains all data required at different stages within 
the model. The `output_data` folder contains the result sub-folders. The `src_acquisition` folder contains the script
used for reanalysis data download from the ECMWF ERA5 database. The `src` folder contains the scripts running the `resite`
model. Finally, the `src_postprocessing` folder includes scripts used for analysing results.

#### I. Resource data retrieval
It should be noted that usable data (incl. resource data, land characteristics, etc.) are already available 
[here](https://dox.uliege.be/index.php/s/trys2xY7j9JsQ3z). We recommend using this data to get a good understanding
of the model architecture and cababilities. In case additional data is needed, the acquisition procedure detailed below
should suffice. 
1. Set up access to the CDS API by following instructions [here](https://cds.climate.copernicus.eu/api-how-to).
2. Open the `data_retrieval.py` file in `src_acquisition` using any file editor or IDE available and adjust the
download settings as required (e.g., area, time coverage, spatial resolution, variables). 
3. Run `data_retrieval.py`. This will commence a connection to the CDS servers and data
download will start. According to the amount of data requested, this process takes time.
Downloaded files will be saved in the `input_data/resource_data` folder in a subfolder named after the spatial resolution chosen.
                 
#### II. Optimal deployment of generation sites
1. Open the `parameters.yml` file using any available text editor or IDE and adjust the model parameters according to 
the needs. Further comments on how to set these are provided in the file. Save the file.
2. Run `main.py`. Upon completion, the output is saved in a dedicated sub-folder (named based on its creation time stamp) in `output_data`.

#### III. Results assessment
Run `output.py` in `src_postprocessing`. In console, define an instance of the `Output` class associated to a run, e.g.
`20190101_123000`, as such:

```
instance = Output('20190101_123000')
```

The `instance` has the following attributes:
+ `return_numerics()`, which returns a series of scalar indicators associated with i) the optimal deployment scheme and
ii) a couple other siting strategies. It should be called as:

```
results = instance.return_numerics(**kwargs)
```
+ `plot_numerics()`, which provides a visual representation of the results obtained above. It is called as:
```
instance.plot_numerics(results, **kwargs)
```
+ `optimal_location_plot()`, which plots the optimal sites on a map. It is called as:
```
instance.optimal_locations_plot()
```
+ `optimal_locations_plot_heatmaps()`, which plots RES capacities per node (if capacities are considered)
```
instance.optimal_locations_plot_heatmaps()
```
    
# Citing
Please cite the following paper if you use this software in your research.

+ D. Radu, M. Berger, R. Fonteneau, A. Dubois, H. Pandžić, Y. Dvorkin, Q. Louveaux, D. Ernst, [Resource
Complementarity as Criterion for Siting Renewable Energy Generation Assets](https://orbi.uliege.be/handle/2268/240014), pre-print, 2019.

# License
This software is released as free software under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.en.html).


