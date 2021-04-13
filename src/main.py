import pickle
import yaml
from os.path import join
from numpy import array
from pyomo.opt import SolverFactory
import time

from helpers import read_inputs, init_folder, custom_log, xarray_to_ndarray, generate_jl_input, get_deployment_vector
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping, retrieve_location_dict, retrieve_site_data
from models import build_ip_model

if __name__ == '__main__':

    custom_log(' Starting data pre-processing')

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
    site_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)

    truncated_data = selected_data(database, site_coordinates, time_horizon)
    capacity_factors_data = return_output(truncated_data, data_path)
    time_windows_data = resource_quality_mapping(capacity_factors_data, siting_parameters)

    criticality_data = xarray_to_ndarray(critical_window_mapping(time_windows_data, model_parameters))
    site_positions = sites_position_mapping(time_windows_data)

    import pickle
    pickle.dump(site_coordinates, open(join(data_path, 'input/site_coordinates.p'), 'wb'), protocol=4)
    import sys
    sys.exit()

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

            seed = 1  # for folder naming purposes only
            for i in range(jl_selected.shape[0]):

                output_folder = init_folder(model_parameters, c, suffix='_MIRSA_seed' + str(seed))
                seed += 1

                with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
                    yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
                with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
                    yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

                jl_selected_seed = jl_selected[i, :]
                jl_objective_seed = jl_objective[i]

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

    else:
        raise ValueError(' This solution method is not available. ')
