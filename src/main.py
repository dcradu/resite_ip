import yaml
import julia
from os.path import join, isdir
from os import makedirs
from numpy import argmax, ceil, float64
import pickle
from time import strftime

from copy import deepcopy
from helpers import read_inputs, xarray_to_ndarray, generate_jl_input, get_potential_per_site, capacity_to_cardinality
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping, retrieve_location_dict, retrieve_site_data

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    logger.info('Starting data pre-processing.')

    model_parameters = read_inputs(f"../config_model.yml")
    siting_parameters = model_parameters['siting_params']
    tech_parameters = read_inputs('../config_techs.yml')

    data_path = model_parameters['data_path']
    spatial_resolution = model_parameters['spatial_resolution']
    time_horizon = model_parameters['time_slice']

    database = read_database(data_path, spatial_resolution)
    site_coordinates, legacy_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)
    truncated_data = selected_data(database, site_coordinates, time_horizon)
    capacity_factors_data = return_output(truncated_data, data_path)

    resampled_data = deepcopy(capacity_factors_data)
    rate = model_parameters['resampling_rate']
    for region in capacity_factors_data.keys():
        for tech in capacity_factors_data[region].keys():
            resampled_data[region][tech] = \
                capacity_factors_data[region][tech].resample(time=f"{rate}H").mean(dim='time')

    time_windows_data = resource_quality_mapping(capacity_factors_data, siting_parameters)
    site_positions = sites_position_mapping(time_windows_data)
    deployment_dict = capacity_to_cardinality(database, model_parameters, tech_parameters, site_coordinates,
                                              legacy_coordinates)
    site_potential_data = get_potential_per_site(time_windows_data, tech_parameters, spatial_resolution)
    criticality_data = xarray_to_ndarray(critical_window_mapping(time_windows_data, site_potential_data,
                                                                 deployment_dict, model_parameters))

    jl_dict = generate_jl_input(deployment_dict, site_coordinates, site_positions, legacy_coordinates)
    total_no_locs = sum(deployment_dict[r][t] for r in deployment_dict.keys() for t in deployment_dict[r].keys())
    c = int(ceil(siting_parameters['c'] * total_no_locs))

    output_dir = join(data_path, f"output/{strftime('%Y%m%d_%H%M%S')}/")
    if not isdir(output_dir):
        makedirs(output_dir)

    with open(join(output_dir, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_dir, 'config_techs.yaml'), 'w') as outfile:
        yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

    logger.info('Data pre-processing finished. Opening Julia instance.')

    j = julia.Julia(compiled_modules=False)
    from julia import Main
    Main.include("jl/SitingHeuristics.jl")

    params = siting_parameters['solution_method']
    jl_sel, jl_obj, _ = Main.main_SA(jl_dict['index_dict'],
                                     jl_dict['deployment_dict'],
                                     jl_dict['legacy_site_list'],
                                     criticality_data.astype('float64'), float64(c),
                                     params['neighborhood'], params['initial_temp'],
                                     params['no_iterations'], params['no_epochs'], params['no_runs'])
    logger.info('Siting heuristics done. Writing results to disk.')

    jl_objective_pick = argmax(jl_obj)
    jl_locations_vector = jl_sel[jl_objective_pick, :]

    locations_dict = retrieve_location_dict(jl_locations_vector, model_parameters, site_positions)
    retrieve_site_data(model_parameters, capacity_factors_data, criticality_data, deployment_dict,
                       site_positions, locations_dict, legacy_coordinates, output_dir, benchmark="PROD")

    pickle.dump(jl_sel, open(join(output_dir, 'solution_matrix.p'), 'wb'))
    pickle.dump(jl_obj, open(join(output_dir, 'objective_vector.p'), 'wb'))

    logger.info(f"Results written to {output_dir}")
