from pyomo.opt import SolverFactory
from shutil import copy
from os.path import join
import pickle
from tqdm import tqdm
from src.helpers import read_inputs, init_folder, custom_log
from src.models import preprocess_input_data, build_model
from src.tools import new_cost_rule, retrieve_location_dict, retrieve_site_data, retrieve_max_run_criticality, \
    retrieve_lower_bound, retrieve_y_idx, build_init_multiplier, retrieve_next_multiplier, retrieve_upper_bound

def main():

    parameters = read_inputs('../config_model.yml')
    keepfiles = parameters['keep_files']
    output_folder = init_folder(keepfiles)
    copy('../config_model.yml', output_folder)

    solution_method = parameters['solution_method']

    solver = parameters['solver']
    MIPGap = parameters['mipgap']
    TimeLimit = parameters['timelimit']
    Threads = parameters['threads']
    # technologies = parameters['technologies']

    input_dict = preprocess_input_data(parameters)

    # Solver options for the MIP problem
    opt = SolverFactory(solver)
    opt.options['MIPGap'] = MIPGap
    opt.options['Threads'] = Threads
    opt.options['TimeLimit'] = TimeLimit
    # Solver options for the relaxations.
    opt_relaxation = SolverFactory(solver)
    opt_relaxation.options['MIPGap'] = 0.02
    # Solver options for the integer Lagrangian
    opt_integer = SolverFactory(solver)
    opt_integer.options['MIPGap'] = MIPGap
    opt_integer.options['Threads'] = Threads
    opt_integer.options['TimeLimit'] = TimeLimit
    opt_integer.options['MIPFocus'] = 3
    # Solver options for the iterative procedure
    opt_persistent = SolverFactory('gurobi_persistent')
    opt_persistent.options['mipgap'] = 0.02

    if solution_method == 'None':

        custom_log(' BB chosen to solve the IP.')

        instance, indices = build_model(parameters, input_dict, output_folder, write_lp=False)
        custom_log(' Sending model to solver.')

        results = opt.solve(instance, tee=True, keepfiles=False, report_timing=False,
                            logfile=join(output_folder, 'solver_log.log'),)

        comp_location_dict = retrieve_location_dict(instance, parameters, input_dict, indices)
        max_location_dict = retrieve_site_data(parameters, input_dict, output_folder, comp_location_dict)
        no_windows_max = retrieve_max_run_criticality(max_location_dict, input_dict, parameters)
        print('Number of non-critical windows for the MAX case: ', no_windows_max)

    # elif solution_method == 'ASM':
    #
    #     iterations_Lagrange = 500
    #     subgradient = 'Inexact'
    #
    #     # Build PCR to extract i) the set of ys to dualize
    #     instance_pcr = build_model_relaxation(parameters, input_dict, formulation='PartialConvex')
    #
    #     custom_log(' Solving the partial convex relaxation...')
    #     results_pcr = opt_relaxation.solve(instance_pcr,
    #                                         tee=False, keepfiles=False, report_timing=False)
    #
    #     lb = retrieve_lower_bound(input_dict, instance_pcr,
    #                                  method='SimulatedAnnealing', multiprocess=False,
    #                                  N = 1, T_init=100., no_iter = 100, no_epoch = 500)
    #
    #     # Build sets of constraints to keep and dualize, respectively.
    #     ys_dual, ys_keep = retrieve_y_idx(instance_pcr, share_random_keep=0.2)
    #
    #     # Build (random sampling within range) initial multiplier (\lambda_0). Affects the estimation of the first UB.
    #     init_multiplier = build_init_multiplier(ys_dual, range=0.5)
    #
    #     # Build PCLR/ILR within the "persistent" interface.
    #     instance_Lagrange = build_model_relaxation(input_dict, formulation='Lagrangian',
    #                                                     subgradient_method=subgradient,
    #                                                     y_dual=ys_dual, y_keep=ys_keep,
    #                                                     multiplier=init_multiplier)
    #     opt_persistent.set_instance(instance_Lagrange)
    #
    #     custom_log(' Solving the Lagrangian relaxations...')
    #     # Iteratively solve and post-process data.
    #     for i in tqdm(range(1, iterations_Lagrange + 1), desc='Lagrangean Relaxation Loop'):
    #
    #         results_Lagrange = opt_persistent.solve(tee=False, keepfiles=False, report_timing=False)
    #
    #         # Compute next multiplier.
    #         incumbent_multiplier = retrieve_next_multiplier(instance_Lagrange, init_multiplier,
    #                                                            ys_keep, ys_dual, i, iterations_Lagrange,
    #                                                            subgradient, a=0.01, share_of_iter=0.8)
    #
    #         # Replace multiplier, hence the objective.
    #         instance_Lagrange.del_component(instance_Lagrange.objective)
    #         instance_Lagrange.objective = new_cost_rule(instance_Lagrange,
    #                                                        ys_keep, ys_dual,
    #                                                        incumbent_multiplier)
    #         opt_persistent.set_objective(instance_Lagrange.objective)
    #
    #         # Assign new value to init_multiplier to use in the iterative multiplier calculation.
    #         init_multiplier = incumbent_multiplier
    #
    #     # Create nd array from list of 1d multipliers from each iteration.
    #     # multiplier_array = concatenate_and_filter_arrays(multiplier_list, ys_keep)
    #
    #     # Build and solve ILP with the incumbent_multiplier.
    #     instance_integer = build_model_relaxation(input_dict, formulation='Lagrangian',
    #                                          subgradient_method='Exact', y_dual=ys_dual, y_keep=ys_keep)
    #
    #     custom_log(' Solving the integer Lagrangian problem...')
    #     results = opt_integer.solve(instance_integer, logfile=join(output_folder, 'solver_log.log'),
    #                                                             tee=True, keepfiles=False, report_timing=False)
    #
    #     custom_log(' UB = {}, LB = {}, gap = {}%'.format(retrieve_upper_bound(results),
    #                                                        lb,
    #                                                        round((retrieve_upper_bound(results) - lb) / lb * 100., 2)))

        # optimal_locations = tl.retrieve_optimal_locations(instance_integer, input_dict['critical_window_matrix'],
        #                                                   technologies, problem)
    #
    #
    #
    # elif solution_method == 'Heuristic':
    # #TODO: set this up.
    #     pass
    #
    # elif solution_method == 'Warmstart':
    # #TODO: set this up.
    #     pass
    #
    # else:
    #
    #     raise ValueError(' This solution method is not available. Retry.')
    #
    #
    # output_dict = {k: v for k, v in input_dict.items() if k in ('region_list', 'coordinates_dict', 'technologies',
    #                                                             'capacity_factors_dict', 'critical_window_matrix',
    #                                                             'capacity_potential_per_node')}
    #
    # output_dict.update({'optimal_location_dict': optimal_locations})
    #
    # if (problem == 'Covering' and 'capacities' in objective) or (problem == 'Load' and objective == 'following'):
    #     deployed_capacities = tl.retrieve_deployed_capacities(instance, technologies,
    #                                                           input_dict['capacity_potential_per_node'])
    #     output_dict.update({'deployed_capacities_dict': deployed_capacities})
    #
    # pickle.dump(output_dict, open(join(output_folder, 'output_model.p'), 'wb'))
    #
    # tl.remove_garbage(keepfiles, output_folder)


if __name__ == "__main__":

    main()