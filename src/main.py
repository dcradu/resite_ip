import pickle
from os.path import join
from shutil import copy

import numpy as np
from pyomo.opt import SolverFactory

from src.helpers import read_inputs, init_folder, custom_log, remove_garbage, generate_jl_output
from src.models import preprocess_input_data, build_model
from src.tools import retrieve_location_dict, retrieve_site_data, retrieve_max_run_criticality, \
    retrieve_location_dict_jl, retrieve_index_dict

parameters = read_inputs('../config_model.yml')
keepfiles = parameters['keep_files']
output_folder = init_folder(keepfiles)
copy('../config_model.yml', output_folder)

solution_method = parameters['solution_method']

solver = parameters['solver']
MIPGap = parameters['mipgap']
TimeLimit = parameters['timelimit']
Threads = parameters['threads']

input_dict = preprocess_input_data(parameters)

# generate_jl_output(parameters['deployment_vector'],
#                    input_dict['criticality_data'],
#                    input_dict['coordinates_data'],
#                    output_folder,
#                    'offshore_country_5y_min')
# import sys
# sys.exit()

if solution_method == 'BB':

    custom_log(' BB chosen to solve the IP.')

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
    print(comp_location_dict)
    max_location_dict = retrieve_site_data(parameters, input_dict, output_folder, comp_location_dict)
    no_windows_max = retrieve_max_run_criticality(max_location_dict, input_dict, parameters)
    print('Number of non-critical windows for the MAX case: ', no_windows_max)

    input_dict.update({'comp_location_dict': comp_location_dict})
    input_dict.update({'max_location_dict': max_location_dict})
    input_dict.update({'region_list': parameters['regions']})
    input_dict.update({'technologies': parameters['technologies']})

    pickle.dump(input_dict, open(join(output_folder, 'output_model.p'), 'wb'))

    remove_garbage(keepfiles, output_folder)

elif solution_method == 'jl_test':

    _, _, _, indices = retrieve_index_dict(parameters, input_dict['coordinates_data'])

    jl_dict = generate_jl_output(parameters['deployment_vector'], input_dict['criticality_data'], input_dict['coordinates_data'])

    import julia
    from julia.api import Julia
    print('Opening Julia instance')
    jl = julia.Julia(compiled_modules=False)
    fn = jl.include("jl/main_heuristics.jl")

    cs = [8, 12, 16, 20, 38]
    for c in cs:
        print('c:', c)
        jl_selected = fn(jl_dict['index_dict'], jl_dict['deployment_dict'], jl_dict['criticality_matrix'],
                         c, 1, 150, 500, 200, 1, "SALS")
        jl_locations = retrieve_location_dict_jl(jl_selected, parameters, input_dict, indices)

        retrieve_site_data(parameters, input_dict, output_folder, jl_locations,
                           suffix='_'+str(parameters['norm_type'])+'_n38_c'+str(c))

    # pickle.dump(output_dict, open(join(output_folder, str(name) + '.p'), 'wb'), protocol=-1)

    # retrieve_site_data_jl()

# elif solution_method == 'ASM':
#
# 1) generate_jl_output
# 2) subprocess.call (julia, heuristic, file_name)
# 3) init warm solution
# 4) subprocess.call (gurobi_cl, solver_options)
# 5) retrieve_location_dict, retrieve_site_data, retrieve_max_run_criticality

else:

    raise ValueError(' This solution method is not available. Retry.')
