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

    # Read RES dataset and filter coordinates
    database = read_database(data_path, spatial_resolution)

    if isfile(join(data_path, "input/capacity_factors_data.p")):

        capacity_factors_data = pickle.load(open(join(data_path, "input/capacity_factors_data.p"), 'rb'))
        site_coordinates = pickle.load(open(join(data_path, "input/site_coordinates.p"), 'rb'))
        legacy_coordinates = pickle.load(open(join(data_path, "input/legacy_coordinates.p"), 'rb'))
        logger.info('Input files read from disk.')

    else:

        site_coordinates, legacy_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)
        # Truncate dataset based on temporal horizon and locations of interest
        truncated_data = selected_data(database, site_coordinates, time_horizon)
        # Convert resource data into capacity factors
        capacity_factors_data = return_output(truncated_data, data_path)

        pickle.dump(capacity_factors_data, open(join(data_path, f"input/capacity_factors_data.p"), 'wb'), protocol=4)
        pickle.dump(site_coordinates, open(join(data_path, f"input/site_coordinates.p"), 'wb'), protocol=4)
        pickle.dump(legacy_coordinates, open(join(data_path, f"input/legacy_coordinates.p"), 'wb'), protocol=4)
        logger.info('Input files written to disk.')

    site_positions = sites_position_mapping(capacity_factors_data)
    # Convert capacities to cardinality constraints
    deployment_dict = capacity_to_cardinality(database, model_parameters, tech_parameters,
                                              site_coordinates, legacy_coordinates)
    # Get technical potential for all candidate sites
    site_potential_data = get_potential_per_site(capacity_factors_data, tech_parameters, spatial_resolution)

    jl_dict = generate_jl_input(deployment_dict, site_coordinates, site_positions, legacy_coordinates)
    logger.info('Data pre-processing finished. Opening Julia instance.')

    j = julia.Julia(compiled_modules=False)
    from julia import Main
    Main.include("jl/SitingHeuristics.jl")

    siting_parameters = model_parameters['siting_params']
    if siting_parameters['CRIT']['set']:
        output_dir = f"{data_path}/output/BOOK/CRIT_c{siting_parameters['CRIT']['c']}_ls0.15/"
        if not isdir(output_dir):
            makedirs(output_dir)

        # Update the temporal index from time instants to time windows
        time_windows_data = resource_quality_mapping(capacity_factors_data, siting_parameters['CRIT'])
        # Compute criticality (binary) matrix
        criticality_data = xarray_to_ndarray(critical_window_mapping(time_windows_data, site_potential_data,
                                                                     deployment_dict, model_parameters))
        c = int(ceil(siting_parameters['CRIT']['c'] *
                     sum(deployment_dict[r][t] for r in deployment_dict.keys() for t in deployment_dict[r].keys())))

        params = siting_parameters['CRIT']['solution_method']
        jl_sel, jl_obj = Main.main_DGH(jl_dict['index_dict'],
                                       jl_dict['deployment_dict'],
                                       jl_dict['legacy_site_list'],
                                       criticality_data.astype('float64'), float64(c),
                                       params['no_runs'], params['algorithm'])
        jl_objective_pick = argmax(jl_obj)
        jl_selected = jl_sel[jl_objective_pick, :]

    elif siting_parameters['MUSS']['set']:
        output_dir = f"{data_path}/output/BOOK/MUSS_{siting_parameters['MUSS']['alpha']}_{siting_parameters['MUSS']['biobj']}/"
        if not isdir(output_dir):
            makedirs(output_dir)

        # Compute maximum theoretical potential time series
        generation_data = xarray_to_ndarray(power_output_mapping(capacity_factors_data, site_potential_data))
        # Compute load data
        load_data = load_data_mapping(data_path, model_parameters)

        # Run julia code
        params = siting_parameters['MUSS']
        jl_selected = Main.main_MUSS(generation_data, load_data, jl_dict['index_dict'],
                                     jl_dict['deployment_dict'], params['alpha'], params['biobj'],
                                     params['tau'], params['algorithm'])

    elif siting_parameters['CORR']['set']:
        output_dir = f"{data_path}/output/BOOK/CORR/"
        if not isdir(output_dir):
            makedirs(output_dir)

        # Compute covariance matrix
        correlation_data = return_correlation_matrix(capacity_factors_data)
        # Run julia code
        jl_selected = Main.main_CORR(correlation_data, jl_dict['deployment_dict'], jl_dict['index_dict'])

    elif siting_parameters['PROD']['set']:
        output_dir = f"{data_path}/output/BOOK/PROD/"
        if not isdir(output_dir):
            makedirs(output_dir)

    if siting_parameters['PROD']['set']:
        locations_dict = retrieve_prod_sites(model_parameters, capacity_factors_data,
                                             deployment_dict, legacy_coordinates)
    else:
        locations_dict = retrieve_location_dict(jl_selected, model_parameters, site_positions)
    print(locations_dict)
    retrieve_site_data(model_parameters, capacity_factors_data, locations_dict, output_dir)

    with open(join(output_dir, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_dir, 'config_techs.yaml'), 'w') as outfile:
        yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

    logger.info('Siting heuristics done. Writing results to disk.')
