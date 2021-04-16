import yaml
import julia
from os.path import join
from numpy import argmax

from helpers import read_inputs, init_folder, xarray_to_ndarray, generate_jl_input, get_deployment_vector
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping, retrieve_location_dict, retrieve_site_data

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    logger.info('Starting data pre-processing.')

    model_parameters = read_inputs('../config_model.yml')
    siting_parameters = model_parameters['siting_params']
    tech_parameters = read_inputs('../config_techs.yml')

    data_path = model_parameters['data_path']
    spatial_resolution = model_parameters['spatial_resolution']
    time_horizon = model_parameters['time_slice']
    deployment_dict = get_deployment_vector(model_parameters['regions'],
                                            model_parameters['technologies'],
                                            model_parameters['deployments'])

    database = read_database(data_path, spatial_resolution)
    site_coordinates, legacy_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)

    truncated_data = selected_data(database, site_coordinates, time_horizon)
    capacity_factors_data = return_output(truncated_data, data_path)
    time_windows_data = resource_quality_mapping(capacity_factors_data, siting_parameters)

    criticality_data = xarray_to_ndarray(critical_window_mapping(time_windows_data, model_parameters))
    site_positions = sites_position_mapping(time_windows_data)

    jl_dict = generate_jl_input(deployment_dict, site_coordinates, site_positions, legacy_coordinates)

    logger.info('Data pre-processing finished. Opening Julia instance.')
    j = julia.Julia(compiled_modules=False)
    from julia import Main
    Main.include("jl/SitingHeuristics.jl")

    if siting_parameters['solution_method']['SA']['set']:

        params = siting_parameters['solution_method']['SA']
        logger.info(f"{params['algorithm']}_SA chosen to solve the IP.")

        jl_sel, jl_obj, _ = Main.main_SA(jl_dict['index_dict'],
                                         jl_dict['deployment_dict'],
                                         jl_dict['legacy_site_list'],
                                         criticality_data, siting_parameters['c'],
                                         params['neighborhood'], params['initial_temp'], params['p'],
                                         params['no_iterations'], params['no_epochs'],
                                         params['no_runs'], params['no_runs_init'],
                                         params['algorithm'])

        output_folder = init_folder(model_parameters, siting_parameters['c'], suffix=f"_SA_{params['algorithm']}")

    elif siting_parameters['solution_method']['SGH']['set']:

        params = siting_parameters['solution_method']['SGH']
        logger.info('SGH chosen to solve the IP.')

        jl_sel, jl_obj = Main.main_SGH(jl_dict['index_dict'],
                                       jl_dict['deployment_dict'],
                                       jl_dict['legacy_site_list'],
                                       criticality_data, siting_parameters['c'],
                                       params['p'], params['no_runs'], params['algorithm'])

        output_folder = init_folder(model_parameters, siting_parameters['c'], suffix="_SGH")

    else:
        raise ValueError(' This solution method is not available.')

    with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
        yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

    logger.info('Siting heuristics done. Writing results.')

    jl_objective_pick = argmax(jl_obj)
    jl_locations_vector = jl_sel[jl_objective_pick, :]

    locations_dict = retrieve_location_dict(jl_locations_vector, model_parameters, site_positions)
    retrieve_site_data(model_parameters, capacity_factors_data, criticality_data,
                       site_positions, locations_dict, legacy_coordinates, output_folder, benchmark='PROD')

    logger.info(f"Results written to {output_folder}")
