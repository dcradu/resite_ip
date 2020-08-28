import pickle
import yaml
from os.path import join
from random import sample
from pyomo.opt import SolverFactory
from numpy import zeros, argmax
import argparse


from src.helpers import read_inputs, init_folder, custom_log, remove_garbage, generate_jl_output
from src.models import preprocess_input_data, build_model
from src.tools import retrieve_location_dict, retrieve_site_data, retrieve_location_dict_jl, retrieve_index_dict

def parse_args():

    parser = argparse.ArgumentParser(description='Command line arguments.')

    parser.add_argument('-c', '--global_thresh', type=float, help='Global threshold')
    parser.add_argument('-tl', '--time_limit', type=float, help='Solver time limit')
    parser.add_argument('-th', '--threads', type=int, help='Number of threads')

    parsed_args = vars(parser.parse_args())

    return parsed_args

parameters = read_inputs('../config_model.yml')
keepfiles = parameters['keep_files']

input_dict = preprocess_input_data(parameters)

if parameters['solution_method']['BB']['set']:

    args = parse_args()

    parameters['solution_method']['BB']['timelimit'] = args['time_limit']
    parameters['solution_method']['BB']['threads'] = args['thread']
    parameters['solution_method']['BB']['c'] = args['thread']

    custom_log(' BB chosen to solve the IP.')

    solver = parameters['solution_method']['BB']['solver']
    MIPGap = parameters['solution_method']['BB']['mipgap']
    TimeLimit = parameters['solution_method']['BB']['timelimit']
    Threads = parameters['solution_method']['BB']['threads']
    c = parameters['solution_method']['BB']['c']

    if not isinstance(parameters['solution_method']['BB']['c'], int):
        raise ValueError(' Values of c have to be integers for the Branch & Bound set-up.')

    output_folder = init_folder(parameters, input_dict, suffix='_c' + str(parameters['solution_method']['BB']['c']))
    with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

    # Solver options for the MIP problem
    opt = SolverFactory(solver)
    opt.options['MIPGap'] = MIPGap
    opt.options['Threads'] = Threads
    opt.options['TimeLimit'] = TimeLimit

    instance, indices = build_model(parameters, input_dict, output_folder, write_lp=False)
    custom_log(' Sending model to solver.')

    results = opt.solve(instance, tee=True, keepfiles=False, report_timing=False,
                        logfile=join(output_folder, 'solver_log.log'))

    objective = instance.objective()

    comp_location_dict = retrieve_location_dict(instance, parameters, input_dict, indices)
    retrieve_site_data(c, parameters, input_dict, output_folder, comp_location_dict, objective)


elif parameters['solution_method']['RAND']['set']:

    custom_log(' Locations to be chosen via random search.')

    if not isinstance(parameters['solution_method']['RAND']['c'], list):
        raise ValueError(' Values of c have to provided as list for the RAND set-up.')
    if len(parameters['technologies']) > 1:
        raise ValueError(' This method is currently implemented for one single technology only.')

    c = parameters['solution_method']['RAND']['c']
    n, dict_deployment, partitions, indices = retrieve_index_dict(parameters, input_dict['coordinates_data'])

    for c in parameters['solution_method']['RAND']['c']:

        print('Running random search for c value of', c)

        seed = parameters['solution_method']['RAND']['seed']
        for run in range(parameters['solution_method']['RAND']['no_runs']):

            output_folder = init_folder(parameters, input_dict, suffix='_c'+ str(c) + '_rand_seed' + str(seed))
            seed += 1

            with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
                yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

            it = parameters['solution_method']['RAND']['no_iterations']*parameters['solution_method']['RAND']['no_epochs']
            best_objective = 0.
            best_random_locations = []

            for i in range(it):

                random_locations = []
                random_locations_index = []
                all_locations = []

                for region in parameters['deployment_vector'].keys():
                    for tech in parameters['technologies']:

                        population = input_dict['coordinates_data'][region][tech]
                        k = parameters['deployment_vector'][region][tech]

                        all_locations.extend(population)
                        random_locations.extend(sample(population, k))

                for loc in random_locations:
                    idx = all_locations.index(loc)
                    random_locations_index.append(idx)

                random_locations_index = sorted(random_locations_index)

                xs = zeros(shape=input_dict['criticality_data'].shape[1])
                xs[random_locations_index] = 1

                D = input_dict['criticality_data']
                objective = (D.dot(xs) >= c).astype(int).sum()

                if objective > best_objective:
                    best_objective = objective
                    best_random_locations = random_locations
            random_locations_dict = {parameters['technologies'][0]: best_random_locations}
            retrieve_site_data(c, parameters, input_dict, output_folder, random_locations_dict, best_objective)


elif parameters['solution_method']['HEU']['set']:

    custom_log(' HEU chosen to solve the IP. Opening a Julia instance.')
    import julia

    if not isinstance(parameters['solution_method']['HEU']['c'], list):
        raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

    _, _, _, indices = retrieve_index_dict(parameters, input_dict['coordinates_data'])

    jl_dict = generate_jl_output(parameters['deployment_vector'],
                                 input_dict['criticality_data'],
                                 input_dict['coordinates_data'])

    jl = julia.Julia(compiled_modules=False)
    from julia.api import Julia
    fn = jl.include("jl/main_heuristics.jl")

    for c in parameters['solution_method']['HEU']['c']:
        print('Running heuristic for c value of', c)

        jl_selected, jl_objective = fn(jl_dict['index_dict'],
                                       jl_dict['deployment_dict'],
                                       jl_dict['criticality_matrix'],
                                       c,
                                       parameters['solution_method']['HEU']['neighborhood'],
                                       parameters['solution_method']['HEU']['no_iterations'],
                                       parameters['solution_method']['HEU']['no_epochs'],
                                       parameters['solution_method']['HEU']['initial_temp'],
                                       parameters['solution_method']['HEU']['no_runs'],
                                       parameters['solution_method']['HEU']['algorithm'])

        if parameters['solution_method']['HEU']['which_sol'] == 'max':
            jl_objective_seed = max(jl_objective)
            jl_selected_seed = jl_selected[argmax(jl_objective), :]

            output_folder = init_folder(parameters, input_dict, suffix='_c' + str(c))
            with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
                yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

            jl_locations = retrieve_location_dict_jl(jl_selected_seed, parameters, input_dict, indices)
            retrieve_site_data(parameters, input_dict, output_folder, jl_locations, jl_objective_seed)

        else: #'rand'
            seed = parameters['solution_method']['HEU']['seed']
            for i in range(jl_selected.shape[0]):

                output_folder = init_folder(parameters, input_dict, suffix='_c' + str(c) + '_seed' + str(seed))
                seed += 1

                with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
                    yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

                jl_selected_seed = jl_selected[i, :]
                jl_objective_seed = jl_objective[i]

                jl_locations = retrieve_location_dict_jl(jl_selected_seed, parameters, input_dict, indices)
                retrieve_site_data(c, parameters, input_dict, output_folder, jl_locations, jl_objective_seed)

else:
    raise ValueError(' This solution method is not available. Retry.')

remove_garbage(keepfiles, output_folder)

