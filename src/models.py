import tools as tl
from numpy import ones, floor, sum, arange, full, multiply, asarray, dot, delete, array
from pyomo.environ import ConcreteModel, RangeSet, Var, Constraint, Objective, Binary, \
    minimize, maximize, NonNegativeReals, NonNegativeIntegers, Suffix
from pyomo.opt import ProblemFormat, SolverFactory
from os.path import join
from pypsa.opt import l_constraint, LConstraint, l_objective, LExpression

def preprocess_input_data(parameters):
    """Data pre-processing.

    Parameters:

    ------------

    parameters : dict
        Dict containing parameters of the run.


    Returns:

    -----------

    output_dictionary : dict
        Dict containing various data structures.

    """



    region = parameters['regions']
    technologies = parameters['technologies']
    measure = parameters['resource_quality_measure']
    horizon = parameters['time_slice']
    spatial_resolution = parameters['spatial_resolution']

    path_resource_data = parameters['path_resource_data'] + str(spatial_resolution) + '/'
    path_transfer_function_data = parameters['path_transfer_function_data']
    path_population_density_data = parameters['path_population_density_data']
    path_protected_areas_data = parameters['path_protected_areas_data']
    path_landseamask = parameters['path_landseamask']
    path_load_data = parameters['path_load_data']
    path_bus_data = parameters['path_bus_data']

    population_density_threshold = parameters['population_density_threshold']
    protected_areas_selection = parameters['protected_areas_selection']
    threshold_distance = parameters['protected_areas_threshold']
    threshold_depth = parameters['depth_threshold']

    alpha_rule = parameters['alpha_rule']
    alpha_plan = parameters['alpha_plan']
    alpha_load_norm = parameters['alpha_norm']
    alpha_numerical = parameters['alpha_numerical']

    delta = parameters['delta']
    beta = parameters['beta']

    number_of_deployments = parameters['cardinality_constraint']
    capacity_constraint = parameters['capacity_constraint']
    economic_budget = parameters['cost_budget']

    database = tl.read_database(path_resource_data)
    global_coordinates = tl.get_global_coordinates(database,
                                                    spatial_resolution,
                                                    population_density_threshold,
                                                    path_population_density_data,
                                                    protected_areas_selection,
                                                    threshold_distance,
                                                    path_protected_areas_data,
                                                    population_density_layer=False,
                                                    protected_areas_layer=False)
    coordinates_filtered_depth = tl.filter_offshore_coordinates(global_coordinates,
                                                                threshold_depth,
                                                                spatial_resolution,
                                                                path_landseamask)
    region_coordinates = tl.return_coordinates_from_countries(region, coordinates_filtered_depth, add_offshore=True)
    # region_coordinates = tl.return_coordinates(region, coordinates_filtered_depth)

    truncated_data = tl.selected_data(database, region_coordinates, horizon)
    output_data = tl.return_output(truncated_data, technologies, path_transfer_function_data)
    resource_quality = tl.resource_quality_mapping(output_data, delta, measure)

    tl.custom_log(' {} regions {}, {} sites and {} time windows selected.'.format(
                                                    len(region_coordinates),
                                                    region,
                                                    sum(len(region_coordinates[key])
                                                        for key in region_coordinates.keys()),
                                                    resource_quality[technologies[0]].data.shape[0]))

    critical_windows = tl.critical_window_mapping(resource_quality,
                                                  alpha_rule,
                                                  alpha_plan,
                                                  alpha_load_norm,
                                                  alpha_numerical,
                                                  delta,
                                                  region,
                                                  region_coordinates,
                                                  horizon,
                                                  path_load_data)

    partitions, indices = tl.get_indices(technologies, number_of_deployments, region_coordinates)

    load_dict, load_data = tl.read_load_data(path_load_data, horizon)
    load_array, weight_array = tl.build_load_vectors(load_dict, load_data, region)

    distance_array = tl.dist_to_grid_as_penalty(path_bus_data, region_coordinates, technologies)

    capacity_array, cost_array = tl.build_parametric_dataset(
                                            path_landseamask,
                                            path_population_density_data,
                                            path_bus_data,
                                            region_coordinates,
                                            technologies,
                                            spatial_resolution)

    tl.custom_log(' Input data read. Model building starts.')

    output_dictionary = {'coordinates_dict': region_coordinates,
                         'region_list': region,
                         'technologies': technologies,
                         'capacity_factors_dict': output_data,
                         'critical_window_matrix': critical_windows,
                         'partitions': partitions,
                         'location_indices': indices,
                         'number_of_deployments': number_of_deployments,
                         'geographical_coverage': beta,
                         'time_window': delta,
                         'capacity_constraint': capacity_constraint,
                         'economic_budget': economic_budget,
                         'capacity_potential_per_node': capacity_array,
                         'cost_estimation_per_node': cost_array,
                         'distance_from_grid': distance_array,
                         'load_centralized': load_array,
                         'load_weight': weight_array}

    return output_dictionary













def check_model_capacity_cost_feasibility(instance, no_locations, capacity_array, cost_array,
                                          capacity_budget, cost_budget, solver='gurobi'):
    """Building model to check feasibility of the problem for given "b" and "k".

    Parameters:

    ------------

    instance : pyomo.instance

    no_locations : int
        Number of considered locations (for the overall number of techs).

    capacity_array : array
        Array of installed capacity potentials per tech and location.

    cost_array : array
        Array of cost estimates per tech and location.

    capacity_budget : float
        Overall capacity requirements.

    cost_budget : float
        Overall cost constraint.


    Returns:

    -----------

    inner_instance : pyomo_instance

    """


    inner_instance = ConcreteModel()
    inner_instance.L = RangeSet(1, no_locations)
    inner_instance.x = Var(instance.L, within=Binary)

    def capacity_rule(inner_instance):
        return sum(inner_instance.x[l] * capacity_array[l - 1] for l in inner_instance.L) >= capacity_budget

    def objective_rule(inner_instance):
        return sum(inner_instance.x[l] * cost_array[l - 1] for l in inner_instance.L)

    inner_instance.capacity = Constraint(rule=capacity_rule)
    inner_instance.cost = Objective(rule=objective_rule, sense=minimize)

    opt = SolverFactory(solver)
    opt.solve(inner_instance, tee=False, keepfiles=False)

    budget_lower_bound = round(inner_instance.cost(), 1)
    budget_upper_bound = round(sum(cost_array), 1)

    tl.custom_log(' Maximum capacity in the region is: {}'.format(sum(capacity_array)))

    if capacity_budget > sum(capacity_array):
        raise ValueError(' Required capacity is higher than the overall potential.')

    tl.custom_log(' LB, UB on budget are: {}, {}'.format(budget_lower_bound, budget_upper_bound))

    if cost_budget < budget_lower_bound:
        raise ValueError(' Economic budget is too small. Revise.')











def build_model(input_data, problem, objective, output_folder,
                low_memory=False, write_lp=False):
    """Model build-up.

    Parameters:

    ------------

    input_data : dict
        Dict containing various data structures relevant for the run.

    problem : str
        Problem type (e.g., "Covering", "Load-following")

    objective : str
        Objective (e.g., "Floor", "Cardinality", etc.)

    output_folder : str
        Path towards output folder

    low_memory : boolean
        If False, it uses the pypsa framework to build constraints.
        If True, it sticks to pyomo (slower solution).

    write_lp : boolean
        If True, the model is written to an .lp file.


    Returns:

    -----------

    instance : pyomo.instance
        Model instance.

    """

    if problem == 'Covering':

        critical_windows = input_data['critical_window_matrix']
        no_locations = critical_windows.shape[1]
        no_windows = critical_windows.shape[0]

        k = input_data['partitions']

        n = input_data['number_of_deployments']
        indices = input_data['location_indices']
        beta = input_data['geographical_coverage']
        delta = input_data['time_window']

        if not isinstance(delta, int):
            raise ValueError(' Delta has to be an integer in this (covering) setup.')

        distance_array = input_data['distance_from_grid']


    elif problem == 'Load':

        delta = input_data['time_window']

        if not isinstance(delta, list):
            raise ValueError(' Delta has to be a list in this (load-following) setup.')

        output_data = input_data['capacity_factors_dict']
        load_array = input_data['load_centralized']
        critical_windows = input_data['critical_window_matrix']

        u_hat = tl.apply_rolling_transformations(output_data, load_array, delta)['aggregated_capacity_factors']
        load_array_hat = tl.apply_rolling_transformations(output_data, load_array, delta)['aggregated_load_array']

        no_windows = u_hat.shape[0]
        no_locations = u_hat.shape[1]


    else:

        raise ValueError(' This problem does not exist.')

    capacity_array = input_data['capacity_potential_per_node']
    cost_array = input_data['cost_estimation_per_node']
    capacity_budget = input_data['capacity_constraint']
    cost_budget = input_data['economic_budget']





    model = ConcreteModel()

    model.W = arange(1, no_windows + 1)
    model.L = arange(1, no_locations + 1)

    # This is a small workaround for the pypsa-based constraint building.
    no_index = [0]

    if (problem == 'Covering') & (objective == 'cardinality_floor'):

        model.partitions = arange(1, k + 1)

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=Binary)

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]


            model.cardinality_constraint = Constraint(model.partitions,
                                                      rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W,
                                                                 rule=criticality_activation_constraint_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)









    elif (problem == 'Covering') & (objective == 'cardinality_critical'):

        model.partitions = arange(1, k + 1)

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=Binary)

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        no_critwind_per_location = full(no_locations, float(no_windows)) - D.sum(axis=0)
        lamda = 1e-3 / sum(n)
        penalty = multiply(lamda, no_critwind_per_location)

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]


            model.cardinality_constraint = Constraint(model.partitions,
                                                      rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W,
                                                                 rule=criticality_activation_constraint_rule)


            def cost_rule(model):
                return sum(model.y[w] for w in model.W) + sum(model.x[l] * penalty[l - 1] for l in model.L)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression()

            objective.variables.extend([(1, model.y[w]) for w in model.W])
            objective.variables.extend([(penalty[l - 1], model.x[l]) for l in model.L])

            l_objective(model, objective, sense=maximize)







    elif (problem == 'Covering') & (objective == 'cardinality_distance'):

        model.partitions = arange(1, k + 1)

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=Binary)

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        lamda = 1e0
        penalty = asarray(multiply(lamda, distance_array))

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]


            model.cardinality_constraint = Constraint(model.partitions,
                                                      rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W,
                                                                 rule=criticality_activation_constraint_rule)


            def cost_rule(model):
                return sum(model.y[w] for w in model.W) + sum(model.x[l] * penalty[l - 1] for l in model.L)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression()

            objective.variables.extend([(1, model.y[w]) for w in model.W])
            objective.variables.extend([(penalty[l - 1], model.x[l]) for l in model.L])

            l_objective(model, objective, sense=maximize)






    elif (problem == 'Covering') & (objective == 'capacities_max'):

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=Binary)

        check_model_capacity_cost_feasibility(model, no_locations,
                                              capacity_array, cost_array,
                                              capacity_budget, cost_budget)

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        if low_memory == True:

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]

            def minimum_capacity_constraint_rule(model):
                return sum(capacity_array[l - 1] * model.x[l] for l in model.L) >= capacity_budget

            def maximum_cost_constraint_rule(model):
                return sum(cost_array[l - 1] * model.x[l] for l in model.L) <= cost_budget


            model.criticality_activation_constraint = Constraint(model.W,
                                                                 rule=criticality_activation_constraint_rule)
            model.minimum_capacity_constraint = Constraint(rule=minimum_capacity_constraint_rule)
            model.maximum_cost_constraint = Constraint(rule=maximum_cost_constraint_rule)


            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}
            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])
                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            minimum_capacity_constraint = {(item): [[(capacity_array[l - 1], model.x[l])
                                                     for l in model.L], ">=", capacity_budget]
                                           for item in no_index}
            l_constraint(model, "capacity_budget_constraint", minimum_capacity_constraint, no_index)

            maximum_cost_constraint = {(item): [[(cost_array[l - 1], model.x[l])
                                                 for l in model.L], "<=", cost_budget]
                                       for item in no_index}
            l_constraint(model, "cost_budget_constraint", maximum_cost_constraint, no_index)

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)





    elif (problem == 'Covering') & (objective == 'capacities_share'):

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=Binary)
        model.z = Var(model.L, within=NonNegativeReals, bounds=(0,1))

        check_model_capacity_cost_feasibility(model, no_locations,
                                              capacity_array, cost_array,
                                              capacity_budget, cost_budget)

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n) * round((1 - beta), 2)) + 1)

        if low_memory == True:

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]

            def minimum_capacity_constraint_rule(model):
                return sum(capacity_array[l - 1] * model.z[l] for l in model.L) >= capacity_budget

            def maximum_cost_constraint_rule(model):
                return sum(cost_array[l - 1] * model.x[l] +
                           cost_array[l - 1] * model.z[l] for l in model.L) <= cost_budget

            def maximum_capacity_per_location_rule(model, l):
                return model.z[l] <= model.x[l]

            model.criticality_activation_constraint = Constraint(model.W,
                                                                 rule=criticality_activation_constraint_rule)
            model.minimum_capacity_constraint = Constraint(rule=minimum_capacity_constraint_rule)
            model.maximum_cost_constraint = Constraint(rule=maximum_cost_constraint_rule)
            model.maximum_capacity_per_location = Constraint(rule=maximum_capacity_per_location_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}
            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])
                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            minimum_capacity_constraint = {(item): [[(capacity_array[l - 1], model.z[l])
                                                     for l in model.L], ">=", capacity_budget]
                                           for item in no_index}
            l_constraint(model, "capacity_budget_constraint", minimum_capacity_constraint, no_index)

            maximum_cost_constraint = {(item): [[((cost_array[l - 1], model.z[l]),
                                                  (cost_array[l - 1], model.x[l])) for l in model.L],
                                                "<=", cost_budget] for item in no_index}
            l_constraint(model, "cost_budget_constraint", maximum_cost_constraint, no_index)

            maximum_capacity_per_location = {}
            for l in model.L:
                maximum_capacity_per_location[l] = LConstraint(model.z[l], "<=", model.x[l])
            l_constraint(model, "max_capacity_per_location", maximum_capacity_per_location, model.L)

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)








    elif (problem == 'Load') & (objective == 'following'):

        model.x = Var(model.L, within=NonNegativeReals, bounds=(0,1))
        model.y = Var(model.W, within=NonNegativeReals, bounds=(0,1))

        check_model_capacity_cost_feasibility(model, no_locations,
                                              capacity_array, cost_array,
                                              capacity_budget, cost_budget)


        if low_memory == True:

            def criticality_activation_constraint_rule(model, w):
                return sum(u_hat[w - 1, l - 1] * capacity_array[l - 1] * model.x[l] for l in model.L) \
                                                                        >= load_array_hat[w - 1] * model.y[w]

            def maximum_cost_constraint_rule(model):
                return sum(cost_array[l - 1] * model.x[l] for l in model.L) <= cost_budget

            model.criticality_activation_constraint = Constraint(model.W,
                                                                 rule=criticality_activation_constraint_rule)
            model.maximum_cost_constraint = Constraint(rule=maximum_cost_constraint_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}
            power_output = multiply(capacity_array, u_hat)
            for w in model.W:
                lhs = LExpression([(power_output[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(load_array_hat[w - 1], model.y[w])])
                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            maximum_cost_constraint = {(item): [[(cost_array[l - 1], model.x[l])
                                                 for l in model.L], "<=", cost_budget]
                                       for item in no_index}
            l_constraint(model, "cost_budget_constraint", maximum_cost_constraint, no_index)

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)




    else:
        raise ValueError(' This optimization setup is not available yet. Retry.')

    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model















def build_model_relaxation(input_data, formulation,
                           subgradient_method=None, y_dual=None, y_keep=None, multiplier=None,
                           output_folder=None, low_memory=False, write_lp=False):

    critical_windows = input_data['critical_window_matrix']
    no_locations = critical_windows.shape[1]
    no_windows = critical_windows.shape[0]

    k = input_data['partitions']

    n = input_data['number_of_deployments']
    indices = input_data['location_indices']
    beta = input_data['geographical_coverage']

    model = ConcreteModel()

    model.W = arange(1, no_windows + 1)
    model.L = arange(1, no_locations + 1)
    model.partitions = arange(1, k + 1)




    if formulation == 'PartialConvex':

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=NonNegativeReals, bounds=(0,1))

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]

            model.cardinality_constraint = Constraint(model.partitions, rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W, rule=criticality_activation_constraint_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)







    elif formulation == 'PartialConvexPenalty':

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        no_critwind_per_location = full(no_locations, float(no_windows)) - D.sum(axis=0)
        lamda = 1e-2 / sum(n)
        penalty = multiply(lamda, no_critwind_per_location)

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=NonNegativeReals, bounds=(0,1))

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]

            model.cardinality_constraint = Constraint(model.partitions, rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W, rule=criticality_activation_constraint_rule)

            def cost_rule(model):
                sum(model.y[w] for w in model.W) + sum(model.x[l] * penalty[l - 1] for l in model.L)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression()

            objective.variables.extend([(1, model.y[w]) for w in model.W])
            objective.variables.extend([(penalty[l - 1], model.x[l]) for l in model.L])

            l_objective(model, objective, sense=maximize)



    elif formulation == 'PartialConvexInteger':

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        model.x = Var(model.L, within=Binary)
        model.y = Var(model.W, within=NonNegativeIntegers, bounds=(0, model.c))

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.y[w]

            model.cardinality_constraint = Constraint(model.partitions, rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W, rule=criticality_activation_constraint_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)








    elif formulation == 'LP':

        D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        model.x = Var(model.L, within=NonNegativeReals, bounds=(0,1))
        model.y = Var(model.W, within=NonNegativeReals, bounds=(0,1))

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                return sum(D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]

            model.cardinality_constraint = Constraint(model.partitions, rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W, rule=criticality_activation_constraint_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in model.W:
                lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            objective = LExpression([(1, model.y[w]) for w in model.W])
            l_objective(model, objective, sense=maximize)

        model.dual = Suffix(direction=Suffix.IMPORT)








    elif formulation == 'Lagrangian':

        model.D = ones((no_windows, no_locations)) - critical_windows.values
        model.c = int(floor(sum(n)*round((1 - beta), 2)) + 1)

        model.x = Var(model.L, within=Binary)

        if subgradient_method == 'Inexact':
            model.y = Var(model.W, within=NonNegativeReals, bounds=(0, 1))
        elif subgradient_method == 'Exact':
            model.y = Var(model.W, within=Binary)
        else:
            raise ValueError(' This case is not available.')

        if low_memory == True:

            def cardinality_constraint_rule(model, i):
                return sum(model.x[l] for l in indices[i - 1]) == n[i - 1]

            def criticality_activation_constraint_rule(model, w):
                if w in y_keep:
                    return sum(model.D[w - 1, l - 1] * model.x[l] for l in model.L) >= model.c * model.y[w]
                else:
                    return Constraint.Skip

            model.cardinality_constraint = Constraint(model.partitions, rule=cardinality_constraint_rule)
            model.criticality_activation_constraint = Constraint(model.W, rule=criticality_activation_constraint_rule)

            def cost_rule(model):
                return sum(model.y[w] for w in model.W) + \
                       sum(multiplier[w] * (sum(model.D[w - 1, l - 1] * model.x[l] for l in model.L) - model.c * model.y[w]) for w in y_dual)

            model.objective = Objective(rule=cost_rule, sense=maximize)

        else:

            activation_constraint = {}

            for w in y_keep:

                lhs = LExpression([(model.D[w - 1, l - 1], model.x[l]) for l in model.L])
                rhs = LExpression([(model.c, model.y[w])])

                activation_constraint[w] = LConstraint(lhs, ">=", rhs)

            l_constraint(model, "activation_constraint", activation_constraint, y_keep)

            cardinality_constraint = {}

            for i in model.partitions:
                lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
                rhs = LExpression(constant=n[i - 1])

                cardinality_constraint[i] = LConstraint(lhs, "==", rhs)

            l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))

            lc = dict(zip(y_dual, multiply(model.c, array(list(multiplier.values())))))
            dx = dot(array(list(multiplier.values())),
                     delete(model.D, [i - 1 for i in y_keep], axis=0))

            objective = LExpression()
            objective.variables.extend([(1, model.y[w]) for w in model.W])
            objective.variables.extend([(dx[l - 1], model.x[l]) for l in model.L])
            objective.variables.extend([(-lc[w], model.y[w]) for w in y_dual])

            model.objective = Objective(expr=0., sense=maximize)
            model.objective._expr = tl._build_sum_expression(objective.variables)


    else:
        raise ValueError(' This optimization setup is not available yet. Retry.')


    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model