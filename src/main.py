import pickle
import yaml
from os.path import join, isfile
from numpy import array, argsort, sum
from pyomo.opt import SolverFactory
import time

from helpers import read_inputs, init_folder, custom_log, xarray_to_ndarray, generate_jl_input, get_deployment_vector
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping, retrieve_location_dict, retrieve_site_data
from models import build_ip_model

if __name__ == '__main__':

    model_parameters = read_inputs('../config_model.yml')
    siting_parameters = model_parameters['siting_params']
    tech_parameters = read_inputs('../config_techs.yml')

    data_path = model_parameters['data_path']
    spatial_resolution = model_parameters['spatial_resolution']
    time_horizon = model_parameters['time_slice']

    deployment_dict = get_deployment_vector(model_parameters['regions'],
                                            model_parameters['technologies'],
                                            model_parameters['deployments'])

    if isfile(join(data_path, 'input/criticality_matrix.p')):

        custom_log(' WARNING! Instance data read from files.')
        criticality_data = pickle.load(open(join(data_path, 'input/criticality_matrix.p'), 'rb'))
        site_coordinates = pickle.load(open(join(data_path, 'input/site_coordinates.p'), 'rb'))
        capacity_factors_data = pickle.load(open(join(data_path, 'input/capacity_factors_data.p'), 'rb'))
        site_positions = pickle.load(open(join(data_path, 'input/site_positions.p'), 'rb'))

        r = list(site_coordinates.keys())
        d = sum(model_parameters['deployments'])
        t = model_parameters['technologies']
        ts = len(capacity_factors_data[list(site_coordinates.keys())[0]][model_parameters['technologies'][0]].time)
        custom_log(f" Reading data for a model with a spatial resolution of {float(spatial_resolution)}, "
                   f"covering {r}, siting {d} {t} sites and {ts} time steps.")

    else:

        custom_log('Files not available. Starting data pre-processing.')

        database = read_database(data_path, spatial_resolution)
        site_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)
        truncated_data = selected_data(database, site_coordinates, time_horizon)
        capacity_factors_data = return_output(truncated_data, data_path)
        time_windows_data = resource_quality_mapping(capacity_factors_data, siting_parameters)
        criticality_data = xarray_to_ndarray(critical_window_mapping(time_windows_data, model_parameters))
        site_positions = sites_position_mapping(time_windows_data)

        pickle.dump(criticality_data, open(join(data_path, 'input/criticality_matrix.p'), 'wb'), protocol=4)
        pickle.dump(site_coordinates, open(join(data_path, 'input/site_coordinates.p'), 'wb'), protocol=4)
        pickle.dump(capacity_factors_data, open(join(data_path, 'input/capacity_factors_data.p'), 'wb'), protocol=4)
        pickle.dump(site_positions, open(join(data_path, 'input/site_positions.p'), 'wb'), protocol=4)

        custom_log(' Data read. Building model.')

    if siting_parameters['solution_method']['BB']['set']:

        custom_log(' BB chosen to solve the IP.')
        params = siting_parameters['solution_method']['BB']

        if not isinstance(params['c'], int):
            raise ValueError(' Values of c have to be integers for the Branch & Bound set-up.')

        output_folder = init_folder(model_parameters, params['c'], '_BB')
        with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
            yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
        with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
            yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

        # Solver options for the MIP problem
        opt = SolverFactory(params['solver'])
        opt.options['MIPGap'] = params['mipgap']
        opt.options['Threads'] = params['threads']
        opt.options['TimeLimit'] = params['timelimit']

        instance = build_ip_model(deployment_dict, site_coordinates, criticality_data, params['c'], output_folder)
        custom_log(' Sending model to solver.')

        results = opt.solve(instance, tee=True, keepfiles=False,
                            report_timing=False, logfile=join(output_folder, 'solver_log.log'))

        objective = instance.objective()
        x_values = array(list(instance.x.extract_values().values()))
        comp_location_dict = retrieve_location_dict(x_values, model_parameters, site_positions)
        retrieve_site_data(model_parameters, deployment_dict, site_coordinates, capacity_factors_data, criticality_data,
                           site_positions, params['c'], comp_location_dict, objective, output_folder)

    elif siting_parameters['solution_method']['MIRSA']['set']:

        custom_log(' MIRSA chosen to solve the IP. Opening a Julia instance.')
        params = siting_parameters['solution_method']['MIRSA']

        if not isinstance(params['c'], list):
            raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

        jl_dict = generate_jl_input(deployment_dict, site_coordinates)

        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        for c in params['c']:
            print('Running heuristic for c value of', c)
            start = time.time()
            jl_selected, jl_objective, jl_traj = Main.main_MIRSA(jl_dict['index_dict'], jl_dict['deployment_dict'],
                                                                 criticality_data, c, params['neighborhood'],
                                                                 params['no_iterations'], params['no_epochs'],
                                                                 params['initial_temp'], params['no_runs'],
                                                                 params['algorithm'])

            output_folder = init_folder(model_parameters, suffix='_c' + str(c) + '_MIRSA')

            with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
                yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
            with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
                yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

            pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
            pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))
            pickle.dump(jl_traj, open(join(output_folder, 'trajectory_matrix.p'), 'wb'))

            med_run = argsort(jl_objective)[c//2]
            jl_selected_seed = jl_selected[med_run, :]
            jl_objective_seed = jl_objective[med_run]

            jl_locations = retrieve_location_dict(jl_selected_seed, model_parameters, site_positions)
            retrieve_site_data(model_parameters, deployment_dict, site_coordinates, capacity_factors_data,
                               criticality_data, site_positions, c, jl_locations, jl_objective_seed,
                               output_folder, benchmarks=True)

    elif siting_parameters['solution_method']['RAND']['set']:

        custom_log(' Locations to be chosen via random search. Resulting coordinates are not obtained!')
        params = model_parameters['solution_method']['RAND']

        if not isinstance(params['c'], list):
            raise ValueError(' Values of c have to provided as list for the RAND set-up.')
        if len(model_parameters['technologies']) > 1:
            raise ValueError(' This method is currently implemented for one single technology only.')

        jl_dict = generate_jl_input(deployment_dict, site_coordinates)

        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        for c in params['c']:
            print('Running heuristic for c value of', c)

            jl_selected, jl_objective = Main.main_RAND(jl_dict['deployment_dict'], criticality_data,
                                                       c, params['algorithm'])

            output_folder = init_folder(model_parameters, suffix='_c' + str(c) + '_RS')

            pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
            pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

            med_run = argsort(jl_objective)[c//2]
            jl_selected_seed = jl_selected[med_run, :]
            jl_objective_seed = jl_objective[med_run]

            jl_locations = retrieve_location_dict(jl_selected_seed, model_parameters, site_positions)
            retrieve_site_data(model_parameters, deployment_dict, site_coordinates, capacity_factors_data,
                               criticality_data, site_positions, c, jl_locations, jl_objective_seed,
                               output_folder, benchmarks=True)

    elif siting_parameters['solution_method']['GRED']['set']:

        custom_log(' GRED chosen to solve the IP. Opening a Julia instance. Resulting coordinates are not obtained!')
        params = model_parameters['solution_method']['GRED']

        if not isinstance(params['c'], list):
            raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

        jl_dict = generate_jl_input(deployment_dict, site_coordinates)

        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        for c in params['c']:
            print('Running heuristic for c value of', c)
            jl_selected, jl_objective = Main.main_GRED(jl_dict['deployment_dict'], criticality_data, c,
                                                       params['no_runs'], params['eps'], params['algorithm'])

            output_folder = init_folder(model_parameters, suffix='_c' + str(c) + '_GRED')

            pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
            pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

            med_run = argsort(jl_objective)[c//2]
            jl_selected_seed = jl_selected[med_run, :]
            jl_objective_seed = jl_objective[med_run]

            jl_locations = retrieve_location_dict(jl_selected_seed, model_parameters, site_positions)
            retrieve_site_data(model_parameters, deployment_dict, site_coordinates, capacity_factors_data,
                               criticality_data, site_positions, c, jl_locations, jl_objective_seed,
                               output_folder, benchmarks=True)

    else:
        raise ValueError(' This solution method is not available. ')
