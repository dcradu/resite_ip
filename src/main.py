import pickle
import yaml
from os.path import join, isfile
from random import sample
from pyomo.opt import SolverFactory
from numpy import zeros, argmax
import argparse
import time

from helpers import read_inputs, init_folder, custom_log, remove_garbage, generate_jl_output
from models import preprocess_input_data, build_model
from tools import retrieve_location_dict, retrieve_site_data, retrieve_location_dict_jl, retrieve_index_dict

def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('-c', '--global_thresh', type=int, help='Global threshold')
    parser.add_argument('-tl', '--time_limit', type=int, help='Solver time limit')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads')

    parsed_args = vars(parser.parse_args())

    return parsed_args

parameters = read_inputs('../config_model.yml')
keepfiles = parameters['keep_files']

if isfile('../input_data/criticality_matrix.p'):
    print('Files already available.')
    criticality_data = pickle.load(open('../input_data/criticality_matrix.p', 'rb'))
    coordinates_data = pickle.load(open('../input_data/coordinates_data.p', 'rb'))
    output_data = pickle.load(open('../input_data/output_data.p', 'rb'))
    print(' WARNING! Instance data read from files.')
else:
    print('Files not available.')
    input_dict = preprocess_input_data(parameters)
    criticality_data = input_dict['criticality_data']
    coordinates_data = input_dict['coordinates_data']
    output_data = input_dict['capacity_factor_data']

    pickle.dump(input_dict['criticality_data'], open('../input_data/criticality_matrix.p', 'wb'), protocol=4)
    pickle.dump(input_dict['coordinates_data'], open('../input_data/coordinates_data.p', 'wb'), protocol=4)
    pickle.dump(input_dict['capacity_factor_data'], open('../input_data/output_data.p', 'wb'), protocol=4)

if parameters['solution_method']['BB']['set']:

    args = parse_args()

    parameters['solution_method']['BB']['timelimit'] = args['time_limit']
    parameters['solution_method']['BB']['threads'] = args['threads']
    parameters['solution_method']['BB']['c'] = args['global_thresh']

    custom_log(' BB chosen to solve the IP.')

    solver = parameters['solution_method']['BB']['solver']
    MIPGap = parameters['solution_method']['BB']['mipgap']
    TimeLimit = parameters['solution_method']['BB']['timelimit']
    Threads = parameters['solution_method']['BB']['threads']
    c = parameters['solution_method']['BB']['c']

    if not isinstance(parameters['solution_method']['BB']['c'], int):
        raise ValueError(' Values of c have to be integers for the Branch & Bound set-up.')

    output_folder = init_folder(parameters, suffix='_c' + str(parameters['solution_method']['BB']['c']) + '_BB')
    with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

    # Solver options for the MIP problem
    opt = SolverFactory(solver)
    opt.options['MIPGap'] = MIPGap
    opt.options['Threads'] = Threads
    opt.options['TimeLimit'] = TimeLimit

    instance, indices = build_model(parameters, coordinates_data, criticality_data, output_folder, write_lp=False)
    custom_log(' Sending model to solver.')

    results = opt.solve(instance, tee=False, keepfiles=False, report_timing=False,
                        logfile=join(output_folder, 'solver_log.log'))

    objective = instance.objective()
    comp_location_dict = retrieve_location_dict(instance, parameters, coordinates_data, indices)
    retrieve_site_data(c, parameters, coordinates_data, criticality_data, output_data, output_folder, comp_location_dict, objective)

elif parameters['solution_method']['RAND']['set']:

    custom_log(' Locations to be chosen via random search.')

    if not isinstance(parameters['solution_method']['RAND']['c'], list):
        raise ValueError(' Values of c have to provided as list for the RAND set-up.')
    if len(parameters['technologies']) > 1:
        raise ValueError(' This method is currently implemented for one single technology only.')

    import julia
    jl_dict = generate_jl_output(parameters['deployment_vector'],
                                 criticality_data,
                                 coordinates_data)
    jl = julia.Julia(compiled_modules=False)
    from julia.api import Julia
    jl.include("jl/main_heuristics_module.jl")
    from julia import Siting_Heuristics as heuristics

    for c in parameters['solution_method']['RAND']['c']:
        print('Running heuristic for c value of', c)
        start = time.time()
        jl_selected, jl_objective = heuristics.main_RAND(jl_dict['deployment_dict'],
                                                         jl_dict['criticality_matrix'],
                                                         c,
                                                         parameters['solution_method']['RAND']['algorithm'])
        end = time.time()
        noruns = parameters['solution_method']['RAND']['no_runs']
        dt = (end-start)/noruns
        print(f'Average time per run: {dt}')

        output_folder = init_folder(parameters, suffix='_c' + str(c) + '_RS')

        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

elif parameters['solution_method']['MIRSA']['set']:

    custom_log(' HEU chosen to solve the IP. Opening a Julia instance.')
    import julia

    if not isinstance(parameters['solution_method']['MIRSA']['c'], list):
        raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

    jl_dict = generate_jl_output(parameters['deployment_vector'],
                                 criticality_data,
                                 coordinates_data)

    jl = julia.Julia(compiled_modules=False)
    from julia.api import Julia
    jl.include("jl/main_heuristics_module.jl")
    from julia import Siting_Heuristics as heuristics

    for c in parameters['solution_method']['MIRSA']['c']:
        print('Running heuristic for c value of', c)
        start = time.time()
        jl_selected, jl_objective, jl_traj = heuristics.main_MIRSA(jl_dict['index_dict'],
                                             jl_dict['deployment_dict'],
                                             jl_dict['criticality_matrix'],
                                             c,
                                             parameters['solution_method']['MIRSA']['neighborhood'],
                                             parameters['solution_method']['MIRSA']['no_iterations'],
                                             parameters['solution_method']['MIRSA']['no_epochs'],
                                             parameters['solution_method']['MIRSA']['initial_temp'],
                                             parameters['solution_method']['MIRSA']['no_runs'],
                                             parameters['solution_method']['MIRSA']['algorithm'])
        end = time.time()
        noruns = parameters['solution_method']['MIRSA']['no_runs']
        dt = (end-start)/noruns
        print(f'Average time per run: {dt}')

        output_folder = init_folder(parameters, suffix='_c' + str(c) + '_MIRSA')

        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))
        pickle.dump(jl_traj, open(join(output_folder, 'trajectory_matrix.p'), 'wb'))

elif parameters['solution_method']['GRED']['set']:

    custom_log(' GRED chosen to solve the IP. Opening a Julia instance.')
    import julia

    if not isinstance(parameters['solution_method']['GRED']['c'], list):
        raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

    jl_dict = generate_jl_output(parameters['deployment_vector'],
                                 criticality_data,
                                 coordinates_data)

    j = julia.Julia(compiled_modules=False)
    fn = j.include("jl/SitingHeuristics_GRED.jl")

    for c in parameters['solution_method']['GRED']['c']:
        print('Running heuristic for c value of', c)
        start = time.time()
        jl_selected, jl_objective = fn(jl_dict['deployment_dict'],
                                       jl_dict['criticality_matrix'],
                                       c,
                                       parameters['solution_method']['GRED']['no_runs'],
                                       parameters['solution_method']['GRED']['eps'],
                                       parameters['solution_method']['GRED']['algorithm'])
        
        end = time.time()
        noruns = parameters['solution_method']['GRED']['no_runs']
        dt = (end - start) / noruns
        print(f'Average time per run: {dt}')

        output_folder = init_folder(parameters, suffix='_c' + str(c) + '_GRED')

         with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
             yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)
        pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
        pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))
        # if c == parameters['solution_method']['HEU']['c'][0]:
        #     pickle.dump(input_dict['criticality_data'], open(join(output_folder, 'criticality_matrix.p'), 'wb'),
        #                 protocol=4)

else:
    raise ValueError(' This solution method is not available. Retry.')

remove_garbage(keepfiles, output_folder)

