import pickle
import yaml
from os.path import join, isfile
from numpy import array
from pyomo.opt import SolverFactory
import time

from src.helpers import read_inputs, init_folder, custom_log, generate_jl_input
from src.models import preprocess_input_data, build_model
from src.tools import retrieve_location_dict, retrieve_site_data

if __name__ == '__main__':

    parameters = read_inputs('../config_model.yml')
    keepfiles = parameters['keep_files']

    if isfile('../input_data/criticality_matrix.p'):
        custom_log(' WARNING! Instance data read from files. Make sure the files are the ones that you need.')
        criticality_data = pickle.load(open('../input_data/criticality_matrix.p', 'rb'))
        coordinates_data_on_loc = pickle.load(open('../input_data/coordinates_data.p', 'rb'))
        output_data = pickle.load(open('../input_data/output_data.p', 'rb'))
        site_positions = pickle.load(open('../input_data/site_positions.p', 'rb'))
    else:
        custom_log('Files not available.')
        input_dict = preprocess_input_data(parameters)
        criticality_data = input_dict['criticality_data']
        coordinates_data_on_loc = input_dict['coordinates_data']
        output_data = input_dict['capacity_factor_data']
        site_positions = input_dict['site_positions_in_matrix']

        pickle.dump(input_dict['criticality_data'], open('../input_data/criticality_matrix.p', 'wb'), protocol=4)
        pickle.dump(input_dict['coordinates_data'], open('../input_data/coordinates_data.p', 'wb'), protocol=4)
        pickle.dump(input_dict['capacity_factor_data'], open('../input_data/output_data.p', 'wb'), protocol=4)
        pickle.dump(input_dict['site_positions_in_matrix'], open('../input_data/site_positions.p', 'wb'), protocol=4)

    if parameters['solution_method']['BB']['set']:

        custom_log(' BB chosen to solve the IP.')
        params = parameters['solution_method']['BB']

        if not isinstance(params['c'], int):
            raise ValueError(' Values of c have to be integers for the Branch & Bound set-up.')

        output_folder = init_folder(parameters, params['c'], '_BB')
        with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
            yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

        # Solver options for the MIP problem
        opt = SolverFactory(params['solver'])
        opt.options['MIPGap'] = params['mipgap']
        opt.options['Threads'] = params['threads']
        opt.options['TimeLimit'] = params['timelimit']

        instance = build_model(parameters, coordinates_data_on_loc, criticality_data, output_folder, write_lp=False)
        custom_log(' Sending model to solver.')

        results = opt.solve(instance, tee=True, keepfiles=False,
                            report_timing=False, logfile=join(output_folder, 'solver_log.log'))

        objective = instance.objective()
        x_values = array(list(instance.x.extract_values().values()))
        comp_location_dict = retrieve_location_dict(x_values, parameters, site_positions)
        retrieve_site_data(parameters, coordinates_data_on_loc, output_data, criticality_data, site_positions,
                           params['c'], comp_location_dict, objective, output_folder)

    elif parameters['solution_method']['MIRSA']['set']:

        import julia
        custom_log(' MIRSA chosen to solve the IP. Opening a Julia instance.')
        params = parameters['solution_method']['MIRSA']

        if not isinstance(params['c'], list):
            raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')

        jl_dict = generate_jl_input(parameters['deployment_vector'], coordinates_data_on_loc)

        j = julia.Julia(compiled_modules=False)
        fn = j.include("jl/SitingHeuristics_MIRSA.jl")

        for c in params['c']:
            print('Running heuristic for c value of', c)
            start = time.time()
            jl_selected, jl_objective, jl_traj = fn(jl_dict['index_dict'], jl_dict['deployment_dict'],
                                                    criticality_data, c, params['neighborhood'], params['no_iterations'],
                                                    params['no_epochs'], params['initial_temp'], params['no_runs'],
                                                    params['algorithm'])

            if params['purpose'] == 'planning':
                seed = params['seed']
                for i in range(jl_selected.shape[0]):

                    output_folder = init_folder(parameters, c, suffix='_MIRSA_seed' + str(seed))
                    seed += 1

                    with open(join(output_folder, 'config_model.yaml'), 'w') as outfile:
                        yaml.dump(parameters, outfile, default_flow_style=False, sort_keys=False)

                    jl_selected_seed = jl_selected[i, :]
                    jl_objective_seed = jl_objective[i]

                    jl_locations = retrieve_location_dict(jl_selected_seed, parameters, site_positions)
                    retrieve_site_data(parameters, coordinates_data_on_loc, output_data, criticality_data,
                                       site_positions, c, jl_locations, jl_objective_seed, output_folder)
            else:

                output_folder = init_folder(parameters, suffix='_c' + str(c) + '_MIRSA')

                pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
                pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))
                pickle.dump(jl_traj, open(join(output_folder, 'trajectory_matrix.p'), 'wb'))

    # elif parameters['solution_method']['RAND']['set']:
    #
    #     import julia
    #     custom_log(' Locations to be chosen via random search.')
    #     params = parameters['solution_method']['RAND']
    #
    #     if not isinstance(params['c'], list):
    #         raise ValueError(' Values of c have to provided as list for the RAND set-up.')
    #     if len(parameters['technologies']) > 1:
    #         raise ValueError(' This method is currently implemented for one single technology only.')
    #
    #     jl_dict = generate_jl_input(parameters['deployment_vector'], coordinates_data_on_loc)
    #
    #     j = julia.Julia(compiled_modules=False)
    #     fn = j.include("jl/SitingHeuristics_RAND.jl")
    #
    #     for c in params['c']:
    #         print('Running heuristic for c value of', c)
    #
    #         jl_selected, jl_objective = fn(jl_dict['deployment_dict'], criticality_data, c, params['algorithm'])
    #
    #         output_folder = init_folder(parameters, suffix='_c' + str(c) + '_RS')
    #
    #         pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
    #         pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))
    #
    # elif parameters['solution_method']['GRED']['set']:
    #
    #     import julia
    #     custom_log(' GRED chosen to solve the IP. Opening a Julia instance.')
    #     params = parameters['solution_method']['GRED']
    #
    #     if not isinstance(params['c'], list):
    #         raise ValueError(' Values of c have to elements of a list for the heuristic set-up.')
    #
    #     jl_dict = generate_jl_input(parameters['deployment_vector'], coordinates_data_on_loc)
    #
    #     j = julia.Julia(compiled_modules=False)
    #     fn = j.include("jl/SitingHeuristics_GRED.jl")
    #
    #     for c in params['c']:
    #         print('Running heuristic for c value of', c)
    #         jl_selected, jl_objective = fn(jl_dict['deployment_dict'], criticality_data, c, params['no_runs'],
    #                                        params['eps'], params['algorithm'])
    #
    #         output_folder = init_folder(parameters, suffix='_c' + str(c) + '_GRED')
    #
    #         pickle.dump(jl_selected, open(join(output_folder, 'solution_matrix.p'), 'wb'))
    #         pickle.dump(jl_objective, open(join(output_folder, 'objective_vector.p'), 'wb'))

    else:
        raise ValueError(' This solution method is not available. ')
