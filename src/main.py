import pickle
from os.path import join
from shutil import copy

from pyomo.opt import SolverFactory

from src.helpers import read_inputs, init_folder, custom_log, remove_garbage, generate_jl_output
from src.models import preprocess_input_data, build_model
from src.tools import retrieve_location_dict, retrieve_site_data, retrieve_max_run_criticality, \
    retrieve_location_dict_jl, retrieve_index_dict

parameters = read_inputs('../config_model.yml')
keepfiles = parameters['keep_files']

solution_method = parameters['solution_method']

solver = parameters['solver']
MIPGap = parameters['mipgap']
TimeLimit = parameters['timelimit']
Threads = parameters['threads']

input_dict = preprocess_input_data(parameters)

if solution_method == 'BB':

    custom_log(' BB chosen to solve the IP.')

    if not isinstance(parameters['c'], int):
        raise ValueError(' Values of c have to be integers for the Branch & Bound set-up.')

    output_folder = init_folder(parameters, input_dict, suffix='_c' + str(parameters['c']))
    pickle.dump(parameters, open(join(output_folder, 'config.yaml'), 'wb'))
    # copy('../config_model.yml', output_folder)

    # Solver options for the MIP problem
    opt = SolverFactory(solver)
    opt.options['MIPGap'] = MIPGap
    opt.options['Threads'] = Threads
    opt.options['TimeLimit'] = TimeLimit

    instance, indices = build_model(parameters, input_dict, output_folder, write_lp=False)
    custom_log(' Sending model to solver.')

    results = opt.solve(instance, tee=True, keepfiles=False, report_timing=False,
                        logfile=join(output_folder, 'solver_log.log'))

    comp_location_dict = retrieve_location_dict(instance, parameters, input_dict, indices)
    max_location_dict = retrieve_site_data(parameters, input_dict, output_folder, comp_location_dict)
    no_windows_max = retrieve_max_run_criticality(max_location_dict, input_dict, parameters)
    # print('Number of non-critical windows for the MAX case: ', no_windows_max)

elif solution_method == 'HEU':

    custom_log(' HEU chosen to solve the IP. Opening a Julia instance.')
    import julia

    if not isinstance(parameters['c'], list):
        raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

    _, _, _, indices = retrieve_index_dict(parameters, input_dict['coordinates_data'])

    jl_dict = generate_jl_output(parameters['deployment_vector'],
                                 input_dict['criticality_data'],
                                 input_dict['coordinates_data'])

    jl = julia.Julia(compiled_modules=False)
    from julia.api import Julia
    fn = jl.include("jl/main_heuristics.jl")

    for c in parameters['c']:
        print('Running heuristic for c value of', c)
        output_folder = init_folder(parameters, input_dict, suffix='_c' + str(c))
        pickle.dump(parameters, open(join(output_folder, 'config.yaml'), 'wb'))
        # copy('../config_model.yml', output_folder)

        jl_selected = fn(jl_dict['index_dict'], jl_dict['deployment_dict'], jl_dict['criticality_matrix'], c,
                         parameters['neighborhood'], parameters['no_iterations'], parameters['no_epochs'],
                         parameters['initial_temp'], parameters['no_runs'], parameters['algorithm'])
        jl_locations = retrieve_location_dict_jl(jl_selected, parameters, input_dict, indices)

        retrieve_site_data(parameters, input_dict, output_folder, jl_locations,
                           suffix=None)

else:
    raise ValueError(' This solution method is not available. Retry.')

remove_garbage(keepfiles, output_folder)

