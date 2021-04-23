import yaml
import julia
from os.path import join, isfile
from numpy import argmax, ceil, float64
import argparse
import pickle

from helpers import read_inputs, init_folder, xarray_to_ndarray, generate_jl_input, \
    get_potential_per_site, capacity_to_cardinality
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping, retrieve_location_dict, retrieve_site_data

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('--c', type=float)
    parser.add_argument('--LS_init_algorithm', type=str, default=None)
    parser.add_argument('--alpha_method', type=str, default=None)
    parser.add_argument('--alpha_coverage', type=str, default=None)
    parser.add_argument('--alpha_norm', type=str, default=None)
    parser.add_argument('--delta', type=int, default=None)

    parsed_args = vars(parser.parse_args())

    return parsed_args


if __name__ == '__main__':

    args = parse_args()

    logger.info('Starting data pre-processing.')

    model_parameters = read_inputs('../config_model.yml')
    siting_parameters = model_parameters['siting_params']
    tech_parameters = read_inputs('../config_techs.yml')

    siting_parameters['alpha']['method'] = args['alpha_method']
    siting_parameters['alpha']['coverage'] = args['alpha_coverage']
    siting_parameters['alpha']['norm'] = args['alpha_norm']
    siting_parameters['delta'] = int(args['delta'])
    siting_parameters['c'] = args['c']
    siting_parameters['solution_method']['SA']['algorithm'] = args['LS_init_algorithm']

    data_path = model_parameters['data_path']
    spatial_resolution = model_parameters['spatial_resolution']
    time_horizon = model_parameters['time_slice']

    database = read_database(data_path, spatial_resolution)

    if isfile(join(data_path, 'input/capacity_factors_data.p')):

        capacity_factors_data = pickle.load(open(join(data_path, 'input/capacity_factors_data_partitioned.p'), 'rb'))
        site_coordinates = pickle.load(open(join(data_path, 'input/site_coordinates_partitioned.p'), 'rb'))
        legacy_coordinates = pickle.load(open(join(data_path, 'input/legacy_coordinates_partitioned.p'), 'rb'))
        logger.info('Input files written to disk.')

    else:

        site_coordinates, legacy_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)
        truncated_data = selected_data(database, site_coordinates, time_horizon)
        capacity_factors_data = return_output(truncated_data, data_path)

        pickle.dump(capacity_factors_data, open(join(data_path, 'input/capacity_factors_data_partitioned.p'), 'wb'), protocol=4)
        pickle.dump(site_coordinates, open(join(data_path, 'input/site_coordinates_partitioned.p'), 'wb'), protocol=4)
        pickle.dump(legacy_coordinates, open(join(data_path, 'input/legacy_coordinates_partitioned.p'), 'wb'), protocol=4)

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

    logger.info('Data pre-processing finished. Opening Julia instance.')

    j = julia.Julia(compiled_modules=False)
    from julia import Main
    Main.include("jl/SitingHeuristics.jl")

    params = siting_parameters['solution_method']['SA']
    logger.info(f"{params['algorithm']}_SA chosen to solve the IP.")

    jl_sel, jl_obj, jl_tra = Main.main_SA(jl_dict['index_dict'],
                                         jl_dict['deployment_dict'],
                                         jl_dict['legacy_site_list'],
                                         criticality_data.astype('float64'), float64(c),
                                         params['neighborhood'], params['initial_temp'], params['p'],
                                         params['no_iterations'], params['no_epochs'],
                                         params['no_runs'], params['no_runs_init'],
                                         params['algorithm'])

    output_folder = init_folder(model_parameters, total_no_locs, c,
                                suffix=f"_SA_{params['algorithm']}_{args['alpha_method']}_{args['alpha_coverage']}"
                                f"_{args['alpha_norm']}_d{args['delta']}")

    with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
        yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

    logger.info('Siting heuristics done. Writing results.')

    jl_objective_pick = argmax(jl_obj)
    jl_locations_vector = jl_sel[jl_objective_pick, :]

    locations_dict = retrieve_location_dict(jl_locations_vector, model_parameters, site_positions)
    retrieve_site_data(model_parameters, capacity_factors_data, criticality_data, deployment_dict,
                       site_positions, locations_dict, legacy_coordinates, output_folder, benchmark='PROD')

    pickle.dump(jl_sel, open(join(output_folder, 'solution_matrix.p'), 'wb'))
    pickle.dump(jl_obj, open(join(output_folder, 'objective_vector.p'), 'wb'))
    try:
        pickle.dump(jl_tra, open(join(output_folder, 'trajectory_matrix.p'), 'wb'))
    except NameError:
        pass

    logger.info(f"Results written to {output_folder}")
