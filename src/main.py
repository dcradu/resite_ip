import pickle
import yaml
from os.path import join, isfile
from numpy import array, sum
from pyomo.opt import SolverFactory
import time
import argparse

from helpers import read_inputs, init_folder, custom_log, xarray_to_ndarray, generate_jl_input, get_deployment_vector
from tools import read_database, return_filtered_coordinates, selected_data, return_output, resource_quality_mapping, \
    critical_window_mapping, sites_position_mapping
from models import build_ip_model


def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('--c', type=int)
    parser.add_argument('--p', type=int, default=None)
    parser.add_argument('--run_BB', type=bool, default=False)
    parser.add_argument('--run_MIR', type=bool, default=False)
    parser.add_argument('--run_LS', type=bool, default=False)
    parser.add_argument('--run_GRED_DET', type=bool, default=False)
    parser.add_argument('--run_GRED_STO', type=bool, default=False)
    parser.add_argument('--run_RAND', type=bool, default=False)
    parser.add_argument('--LS_init_algorithm', type=str, default=None)
    parser.add_argument('--init_sol_folder', type=str, default=None)

    parsed_args = vars(parser.parse_args())

    return parsed_args


def single_true(iterable):
    i = iter(iterable)
    return any(i) and not any(i)


if __name__ == '__main__':

    args = parse_args()

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

    siting_parameters['solution_method']['BB']['set'] = args['run_BB']
    siting_parameters['solution_method']['BB']['mir'] = args['run_MIR']
    siting_parameters['solution_method']['LS']['set'] = args['run_LS']
    siting_parameters['solution_method']['GRED_DET']['set'] = args['run_GRED_DET']
    siting_parameters['solution_method']['GRED_STO']['set'] = args['run_GRED_STO']
    siting_parameters['solution_method']['GRED_STO']['p'] = args['p']
    siting_parameters['solution_method']['RAND']['set'] = args['run_RAND']

    c = args['c']

    if not single_true([args['run_BB'], args['run_LS'], args['run_GRED_DET'], args['run_GRED_STO'], args['run_RAND']]):
        raise ValueError(' More than one run selected in the argparser.')

    if siting_parameters['solution_method']['BB']['set']:

        custom_log(' BB chosen to solve the IP.')
        params = siting_parameters['solution_method']['BB']

        output_folder = init_folder(model_parameters, c, f"_BB_MIR_{args['run_MIR']}")
        with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
            yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
        with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
            yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

        # Solver options for the MIP problem
        opt = SolverFactory(params['solver'])
        opt.options['MIPGap'] = params['mipgap']
        opt.options['Threads'] = params['threads']
        opt.options['TimeLimit'] = params['timelimit']

        instance = build_ip_model(deployment_dict, site_coordinates, criticality_data,
                                  c, output_folder, args['run_MIR'])
        custom_log(' Sending model to solver.')

        results = opt.solve(instance, tee=False, keepfiles=False,
                            report_timing=False, logfile=join(output_folder, 'solver_log.log'))

        objective = instance.objective()
        x_values = array(list(instance.x.extract_values().values()))

        pickle.dump(x_values, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

    elif siting_parameters['solution_method']['LS']['set']:

        custom_log(f" LS_{args['LS_init_algorithm']} chosen to solve the IP. Opening a Julia instance.")
        params = siting_parameters['solution_method']['LS']

        jl_dict = generate_jl_input(deployment_dict, site_coordinates)
        path_to_sol = args['init_sol_folder'] + str(args['c']) + '_GRED_STGH_p' + str(args['p'])
        path_to_init_sol_folder = join(data_path, 'output', path_to_sol)

        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        start = time.time()
        jl_selected, jl_objective, jl_traj = Main.main_MIRSA(jl_dict['index_dict'], jl_dict['deployment_dict'],
                                                             criticality_data, c, params['neighborhood'],
                                                             params['no_iterations'], params['no_epochs'],
                                                             params['initial_temp'], params['no_runs'],
                                                             args['LS_init_algorithm'],
                                                             args['p'], path_to_init_sol_folder)
        end = time.time()
        print(f"Average CPU time for c={c}: {round((end-start)/params['no_runs'], 1)} s")

        output_folder = init_folder(model_parameters, c, suffix=f"_LS_{args['LS_init_algorithm']}")

        with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
            yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
        with open(join(output_folder, 'config_techs.yaml'), 'w') as outfile:
            yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))
        pickle.dump(jl_traj, open(join(output_folder, 'trajectory_matrix.p'), 'wb'))

    elif siting_parameters['solution_method']['RAND']['set']:
    
        custom_log(' Locations to be chosen via random search.')
        params = siting_parameters['solution_method']['RAND']
    
        jl_dict = generate_jl_input(deployment_dict, site_coordinates)
    
        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        jl_selected, jl_objective = Main.main_RAND(jl_dict['deployment_dict'], criticality_data,
                                                   c, params['no_iterations'], params['no_runs'],
                                                   params['algorithm'])
    
        output_folder = init_folder(model_parameters, c, suffix='_RS')
    
        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

    elif siting_parameters['solution_method']['GRED_DET']['set']:

        params = siting_parameters['solution_method']['GRED_DET']
        custom_log(f" GRED_{params['algorithm']} chosen to solve the IP. Opening a Julia instance.")

        jl_dict = generate_jl_input(deployment_dict, site_coordinates)

        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        start = time.time()
        jl_selected, jl_objective = Main.main_GRED(jl_dict['deployment_dict'], criticality_data, c,
                                                   params['no_runs'], params['p'], params['algorithm'])
        end = time.time()
        print(f"Average CPU time for c={c}: {round((end-start)/params['no_runs'], 1)} s")

        output_folder = init_folder(model_parameters, c, suffix=f"_GRED_{params['algorithm']}")

        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

    elif siting_parameters['solution_method']['GRED_STO']['set']:

        params = siting_parameters['solution_method']['GRED_STO']
        custom_log(f" GRED_{params['algorithm']} chosen to solve the IP. Opening a Julia instance.")

        jl_dict = generate_jl_input(deployment_dict, site_coordinates)

        import julia
        j = julia.Julia(compiled_modules=False)
        from julia import Main
        Main.include("jl/SitingHeuristics.jl")

        start = time.time()
        jl_selected, jl_objective = Main.main_GRED(jl_dict['deployment_dict'], criticality_data, c,
                                                   params['no_runs'], params['p'], params['algorithm'])
        end = time.time()
        print(f"Average CPU time for c={c}: {round((end-start)/params['no_runs'], 1)} s")

        output_folder = init_folder(model_parameters, c, suffix=f"_GRED_{params['algorithm']}_p{params['p']}")

        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

    else:
        raise ValueError(' This solution method is not available. ')
