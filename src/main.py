import sys
import models as gg
import tools as tl
from pyomo.opt import SolverFactory
from shutil import copy
from os.path import join
import pickle
from tqdm import tqdm

sys.tracebacklimit = 100

def main():

    parameters = tl.read_inputs('parameters.yml')

    keepfiles = parameters['keep_files']
    lowmem = parameters['low_memory']

    output_folder = tl.init_folder(keepfiles, 'O')

    # Copy parameter file to the output folder for future reference.
    copy('parameters.yml', output_folder)

    problem = parameters['main_problem']
    objective = parameters['main_objective']
    solution_method = parameters['solution_method']
    iterations_Lagrange = parameters['no_iterations_Lagrange']
    iterations_SA = parameters['no_iterations_SA']
    explorations_SA = parameters['no_explorations_SA']
    subgradient = parameters['subgradient_approximation']
    solver = parameters['solver']
    MIPGap = parameters['mipgap']

    technologies = parameters['technologies']

    input_dict = gg.preprocess_input_data(parameters)

    if solver == 'gurobi':
        opt = SolverFactory(solver)
        opt.options['MIPGap'] = MIPGap
        opt.options['Threads'] = 0
        opt.options['TimeLimit'] = 3600
        opt.options['DisplayInterval'] = 600

    elif solver == 'cplex':
        opt = SolverFactory(solver)
        opt.options['mipgap'] = MIPGap
        opt.options['threads'] = 0
        opt.options['mip_limits_treememory'] = 1e3

    # Solver options for the relaxations.
    opt_relaxation = SolverFactory(solver)
    opt_relaxation.options['MIPGap'] = 0.02
    # Solver options for the integer Lagrangian
    opt_integer = SolverFactory(solver)
    opt_integer.options['MIPGap'] = MIPGap
    opt_integer.options['Threads'] = 0
    opt_integer.options['TimeLimit'] = 18000
    opt_integer.options['MIPFocus'] = 3
    opt_integer.options['DisplayInterval'] = 600

    # Solver options for the iterative procedure
    opt_persistent = SolverFactory('gurobi_persistent')
    opt_persistent.options['mipgap'] = 0.02

    if solution_method == 'None':

        if problem == 'Covering':

            instance = gg.build_model(input_dict, problem, objective, output_folder,
                                      low_memory=lowmem, write_lp=False)
            tl.custom_log(' Sending model to solver.')

            results = opt.solve(instance, tee=True, keepfiles=False, report_timing=False,
                                logfile=join(output_folder, 'solver_log.log'),)
            optimal_locations = tl.retrieve_optimal_locations(instance, input_dict['critical_window_matrix'],
                                                              technologies, problem)

        elif problem == 'Load':

            raise ValueError(' This problem should be solved with a solution method.')

        else:

            raise ValueError(' This problem does not exist.')








    elif solution_method == 'Projection':

        if problem == 'Covering':

            instance = gg.build_model_relaxation(input_dict, formulation='PartialConvex', low_memory=lowmem)

        elif problem == 'Load':

            instance = gg.build_model(input_dict, problem, objective, output_folder,
                                      low_memory=lowmem)

        else:

            raise ValueError(' This problem does not exist.')

        tl.custom_log(' Solving...')
        results = opt.solve(instance, logfile=join(output_folder, 'solver_log.log'),
                            tee=True, keepfiles=False, report_timing=False)

        tl.custom_log(' Relaxation solved and passed to the projection problem.')
        tl.retrieve_feasible_solution_projection(input_dict, instance, results, problem)

        optimal_locations = tl.retrieve_optimal_locations(instance,
                                                          input_dict['critical_window_matrix'],
                                                          technologies,
                                                          problem)









    elif solution_method == 'Lagrange':

        # Initialize some list objects to be dumped in the output dict.
        # objective_list = []
        # gap_list = []
        # multiplier_list = []

        # Build PCR to extract i) the set of ys to dualize
        instance_pcr = gg.build_model_relaxation(input_dict, formulation='PartialConvex', low_memory=lowmem)

        tl.custom_log(' Solving the partial convex relaxation...')
        results_pcr = opt_relaxation.solve(instance_pcr,
                                            tee=False, keepfiles=False, report_timing=False)

        lb = tl.retrieve_lower_bound(input_dict, instance_pcr,
                                     method='SimulatedAnnealing', multiprocess=False,
                                     N = 1, T_init=100., no_iter = iterations_SA, no_epoch = explorations_SA)

        # Build sets of constraints to keep and dualize, respectively.
        ys_dual, ys_keep = tl.retrieve_y_idx(instance_pcr, share_random_keep=0.2)

        # Build (random sampling within range) initial multiplier (\lambda_0). Affects the estimation of the first UB.
        init_multiplier = tl.build_init_multiplier(ys_dual, range=0.5)

        # Build PCLR/ILR within the "persistent" interface.
        instance_Lagrange = gg.build_model_relaxation(input_dict, formulation='Lagrangian',
                                                        subgradient_method=subgradient,
                                                        y_dual=ys_dual, y_keep=ys_keep,
                                                        multiplier=init_multiplier, low_memory=lowmem)
        opt_persistent.set_instance(instance_Lagrange)

        tl.custom_log(' Solving the Lagrangian relaxations...')
        # Iteratively solve and post-process data.
        for i in tqdm(range(1, iterations_Lagrange + 1), desc='Lagrangean Relaxation Loop'):

            results_Lagrange = opt_persistent.solve(tee=False, keepfiles=False, report_timing=False)

            # multiplier_list.append(array(list(init_multiplier.values())))
            # objective_list.append(tl.retrieve_upper_bound(results_lagrange))
            # gap_list.append(round((tl.retrieve_upper_bound(results_lagrange) - lb) / lb * 100., 2))

            # Compute next multiplier.
            incumbent_multiplier = tl.retrieve_next_multiplier(instance_Lagrange, init_multiplier,
                                                               ys_keep, ys_dual, i, iterations_Lagrange,
                                                               subgradient, a=0.01, share_of_iter=0.8)

            # Replace multiplier, hence the objective.
            instance_Lagrange.del_component(instance_Lagrange.objective)
            instance_Lagrange.objective = tl.new_cost_rule(instance_Lagrange,
                                                           ys_keep, ys_dual,
                                                           incumbent_multiplier,
                                                           low_memory=lowmem)
            opt_persistent.set_objective(instance_Lagrange.objective)

            # Assign new value to init_multiplier to use in the iterative multiplier calculation.
            init_multiplier = incumbent_multiplier

        # Create nd array from list of 1d multipliers from each iteration.
        # multiplier_array = tl.concatenate_and_filter_arrays(multiplier_list, ys_keep)

        if subgradient == 'Inexact':

            # Build and solve ILP with the incumbent_multiplier.
            instance_integer = gg.build_model_relaxation(input_dict, formulation='Lagrangian',
                                                 subgradient_method='Exact', y_dual=ys_dual, y_keep=ys_keep,
                                                 multiplier=incumbent_multiplier, low_memory=lowmem)

            tl.custom_log(' Solving the integer Lagrangian problem...')
            results = opt_integer.solve(instance_integer, logfile=join(output_folder, 'solver_log.log'),
                                                                    tee=True, keepfiles=False, report_timing=False)

            tl.custom_log(' UB = {}, LB = {}, gap = {}%'.format(tl.retrieve_upper_bound(results),
                                                               lb,
                                                               round((tl.retrieve_upper_bound(results) - lb) / lb * 100., 2)))
            optimal_locations = tl.retrieve_optimal_locations(instance_integer, input_dict['critical_window_matrix'],
                                                              technologies, problem)

        else:

            tl.custom_log(' UB = {}, LB = {}, gap = {}%'.format(tl.retrieve_upper_bound(results_Lagrange),
                                                               lb,
                                                               round((tl.retrieve_upper_bound(results_Lagrange) - lb) / lb * 100., 2)))

            optimal_locations = tl.retrieve_optimal_locations(instance_Lagrange, input_dict['critical_window_matrix'],
                                                              technologies, problem)



    else:

        raise ValueError(' This solution method is not available. Retry.')


    output_dict = {k: v for k, v in input_dict.items() if k in ('region_list', 'coordinates_dict', 'technologies',
                                                                'capacity_factors_dict', 'critical_window_matrix',
                                                                'capacity_potential_per_node')}

    output_dict.update({'optimal_location_dict': optimal_locations})

    if (problem == 'Covering' and 'capacities' in objective) or (problem == 'Load' and objective == 'following'):
        deployed_capacities = tl.retrieve_deployed_capacities(instance, technologies,
                                                              input_dict['capacity_potential_per_node'])
        output_dict.update({'deployed_capacities_dict': deployed_capacities})

    pickle.dump(output_dict, open(join(output_folder, 'output_model.p'), 'wb'))

    tl.remove_garbage(keepfiles, output_folder)


if __name__ == "__main__":

    main()