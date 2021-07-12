import yaml
import julia
from os.path import join, isfile, isdir
from os import makedirs
from numpy import argmax, ceil, float64
import pickle

from helpers import read_inputs, xarray_to_ndarray, generate_jl_input, \
    get_potential_per_site, capacity_to_cardinality, power_output_mapping, load_data_mapping, return_correlation_matrix
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping, retrieve_location_dict, retrieve_site_data, retrieve_prod_sites

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)


if __name__ == '__main__':

    logger.info('Starting data pre-processing.')

    model_parameters = read_inputs(f"../config_model.yml")
    tech_parameters = read_inputs('../config_techs.yml')
    data_path = model_parameters['data_path']
    spatial_resolution = model_parameters['spatial_resolution']
    time_horizon = model_parameters['time_slice']
    siting_criterion = str(model_parameters['criterion'])

    # Read RES dataset and filter coordinates
    database = read_database(data_path, spatial_resolution)

    if isfile(join(data_path, "input/input_chapter_10y/capacity_factors_matrix.p")):

        capacity_factors_data = pickle.load(open(join(data_path, "input/input_chapter_10y/capacity_factors_data.p"), 'rb'))
        capacity_factors_matrix = pickle.load(open(join(data_path, "input/input_chapter_10y/capacity_factors_matrix.p"), 'rb'))
        site_coordinates = pickle.load(open(join(data_path, "input/input_chapter_10y/site_coordinates.p"), 'rb'))
        legacy_coordinates = pickle.load(open(join(data_path, "input/input_chapter_10y/legacy_coordinates.p"), 'rb'))
        demand_vector = pickle.load(open(join(data_path, "input/input_chapter_10y/demand_vector.p"), 'rb'))
        potential_vector = pickle.load(open(join(data_path, "input/input_chapter_10y/potential_vector.p"), 'rb'))
        deployment_dict = pickle.load(open(join(data_path, "input/input_chapter_10y/deployment_dict.p"), 'rb'))
        site_positions = pickle.load(open(join(data_path, "input/input_chapter_10y/site_positions.p"), 'rb'))
        logger.info('Input files read from disk.')

    else:

        site_coordinates, legacy_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)
        # Truncate dataset based on temporal horizon and locations of interest
        truncated_data = selected_data(database, site_coordinates, time_horizon)
        # Convert resource data into capacity factors
        capacity_factors_data = return_output(truncated_data, data_path)

        site_positions = sites_position_mapping(capacity_factors_data)
        # Convert capacities to cardinality constraints
        deployment_dict = capacity_to_cardinality(database, model_parameters, tech_parameters,
                                                  site_coordinates, legacy_coordinates)
        # Get technical potential for all candidate sites
        site_potential_data = get_potential_per_site(capacity_factors_data, tech_parameters, spatial_resolution)

        capacity_factors_matrix = xarray_to_ndarray(capacity_factors_data)
        demand_vector = load_data_mapping(data_path, model_parameters)
        potential_vector = xarray_to_ndarray(site_potential_data)

        pickle.dump(capacity_factors_data, open(join(data_path, f"input/input_chapter_10y/capacity_factors_data.p"), 'wb'), protocol=4)
        pickle.dump(capacity_factors_matrix, open(join(data_path, f"input/input_chapter_10y/capacity_factors_matrix.p"), 'wb'), protocol=4)
        pickle.dump(site_coordinates, open(join(data_path, f"input/input_chapter_10y/site_coordinates.p"), 'wb'), protocol=4)
        pickle.dump(legacy_coordinates, open(join(data_path, f"input/input_chapter_10y/legacy_coordinates.p"), 'wb'), protocol=4)
        pickle.dump(site_positions, open(join(data_path, f"input/input_chapter_10y/site_positions.p"), 'wb'), protocol=4)
        pickle.dump(demand_vector, open(join(data_path, f"input/input_chapter_10y/demand_vector.p"), 'wb'), protocol=4)
        pickle.dump(potential_vector, open(join(data_path, f"input/input_chapter_10y/potential_vector.p"), 'wb'), protocol=4)
        pickle.dump(deployment_dict, open(join(data_path, f"input/input_chapter_10y/deployment_dict.p"), 'wb'), protocol=4)
        logger.info('Input files written to disk.')

    # Get matrix form of all input data.
    deployment_target = sum(deployment_dict[r][t] for r in deployment_dict.keys() for t in deployment_dict[r].keys())
    c = float(ceil(model_parameters['siting_params']['CRIT']['c'] * deployment_target))
    varsigma = float(model_parameters['siting_params']['CRIT']['load_coverage'])

    logger.info('Data pre-processing finished. Opening Julia instance.')

    output_dir = f"{data_path}/output/CHAPTER/{siting_criterion}"
    if not isdir(output_dir):
        makedirs(output_dir)

    j = julia.Julia(compiled_modules=False)
    from julia import Main
    Main.include("jl/SitingGreedy.jl")

    if siting_criterion != "CapacityValue":

        jl_sel, _ = Main.siting_method(capacity_factors_matrix, demand_vector, potential_vector,
                                       deployment_target, c, varsigma, siting_criterion)
        locations_dict = retrieve_location_dict(jl_sel, model_parameters, site_positions)

    else:

        site_potential_data = get_potential_per_site(capacity_factors_data, tech_parameters, spatial_resolution)
        production_data = power_output_mapping(capacity_factors_data, site_potential_data)

        locations_dict = retrieve_prod_sites(model_parameters, production_data, deployment_dict, legacy_coordinates)

    ind_incumbent = retrieve_site_data(model_parameters, capacity_factors_data, locations_dict, output_dir)
    jl_obj = Main.compute_objectives(ind_incumbent, capacity_factors_matrix, demand_vector, potential_vector, deployment_target, c, varsigma)

    with open(join(output_dir, 'objectives.yaml'), 'w') as outfile:
        yaml.dump(jl_obj, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_dir, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_dir, 'config_techs.yaml'), 'w') as outfile:
        yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

    logger.info('Siting heuristics done. Writing results to disk.')
