from os.path import join

from numpy import ones, floor, sum, arange, multiply, dot, delete, array
from pyomo.environ import ConcreteModel, Var, Objective, Binary, \
    maximize, NonNegativeReals
from pyomo.opt import ProblemFormat
from pypsa.opt import l_constraint, LConstraint, l_objective, LExpression, _build_sum_expression

from src.helpers import custom_log, xarray_to_ndarray, concatenate_dict_keys, get_partition_index
from src.tools import read_database, return_filtered_coordinates, selected_data, return_output, \
    resource_quality_mapping, critical_window_mapping


def preprocess_input_data(model_parameters):
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

    regions = model_parameters['deployment_vector'].keys()
    technologies = model_parameters['technologies']

    time_horizon = model_parameters['time_slice']
    spatial_resolution = model_parameters['spatial_resolution']

    path_resource_data = model_parameters['path_resource_data'] + '/' + str(spatial_resolution)
    path_transfer_function_data = model_parameters['path_transfer_function_data']
    path_population_density_data = model_parameters['path_population_density_data']
    path_land_data = model_parameters['path_land_data']
    path_load_data = model_parameters['path_load_data']
    path_legacy_data = model_parameters['path_legacy_data']
    path_shapefile_data = model_parameters['path_shapefile_data']

    resource_quality_layer = model_parameters['resource_quality_layer']
    population_layer = model_parameters['population_density_layer']
    protected_areas_layer = model_parameters['protected_areas_layer']
    bathymetry_layer = model_parameters['bathymetry_layer']
    forestry_layer = model_parameters['forestry_layer']
    orography_layer = model_parameters['orography_layer']
    water_mask_layer = model_parameters['water_mask_layer']
    legacy_layer = model_parameters['legacy_layer']
    latitude_layer = model_parameters['latitude_layer']
    distance_layer = model_parameters['distance_layer']

    delta = model_parameters['delta']
    alpha = model_parameters['alpha']
    measure = model_parameters['smooth_measure']
    norm_type = model_parameters['norm_type']

    database = read_database(path_resource_data)
    filtered_coordinates = return_filtered_coordinates(database, spatial_resolution, technologies, regions,
                                                     path_land_data, path_resource_data, path_legacy_data,
                                                     path_shapefile_data, path_population_density_data,
                                                     resource_quality_layer=resource_quality_layer,
                                                     population_density_layer=population_layer,
                                                     protected_areas_layer=protected_areas_layer,
                                                     orography_layer=orography_layer,
                                                     forestry_layer=forestry_layer,
                                                     water_mask_layer=water_mask_layer,
                                                     bathymetry_layer=bathymetry_layer,
                                                     legacy_layer=legacy_layer,
                                                     latitude_layer=latitude_layer,
                                                     distance_layer=distance_layer)

    truncated_data = selected_data(database, filtered_coordinates, time_horizon)
    output_data = return_output(truncated_data, path_transfer_function_data)

    smooth_data = resource_quality_mapping(output_data, delta, measure)
    critical_data = critical_window_mapping(smooth_data, alpha, delta, regions, time_horizon, path_load_data, norm_type)

    output_dict = {
                'coordinates_data': filtered_coordinates,
                'capacity_factor_data': output_data,
                'criticality_data': xarray_to_ndarray(critical_data)}

    custom_log(' Input data read...')

    return output_dict





def build_model(model_parameters, input_data, output_folder, write_lp=False):
    """Model build-up.

    Parameters:

    ------------


    Returns:

    -----------

    """

    coordinate_dict = input_data['coordinates_data']
    D = input_data['criticality_data']
    no_windows = D.shape[0]
    no_locations = D.shape[1]

    d = model_parameters['deployment_vector']
    if isinstance(d[list(d.keys())[0]], int):
        dict_deployment = d
        n = sum(dict_deployment[item] for item in dict_deployment)
        partitions = [item for item in d]
        if model_parameters['constraint'] == 'country':
            indices = concatenate_dict_keys(get_partition_index(coordinate_dict, d, capacity_split='per_country'))
        elif model_parameters['constraint'] == 'tech':
            indices = concatenate_dict_keys(get_partition_index(coordinate_dict, d, capacity_split='per_tech'))
    else:
        dict_deployment = concatenate_dict_keys(d)
        n = sum(dict_deployment[item] for item in dict_deployment)
        partitions = [item for item in dict_deployment]
        indices = concatenate_dict_keys(get_partition_index(coordinate_dict, d, capacity_split='per_country_and_tech'))

    c = int(floor(n*round((1 - model_parameters['beta']), 2)) + 1)

    for item in partitions:
        if item in indices:
            if dict_deployment[item] > len(indices[item]):
                raise ValueError(' More nodes required than available for {}'.format(item))
        else:
            indices[item] = []
            print('Warning! {} not in keys of choice. Make sure there is no requirement here.'.format(item))

    model = ConcreteModel()

    model.W = arange(1, no_windows + 1)
    model.L = arange(1, no_locations + 1)

    model.x = Var(model.L, within=Binary)
    model.y = Var(model.W, within=Binary)

    activation_constraint = {}

    for w in model.W:
        lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
        rhs = LExpression([(c, model.y[w])])

        activation_constraint[w] = LConstraint(lhs, ">=", rhs)

    l_constraint(model, "activation_constraint", activation_constraint, list(model.W))

    cardinality_constraint = {}

    for item in partitions:
        lhs = LExpression([(1, model.x[l]) for l in indices[item]])
        rhs = LExpression(constant=dict_deployment[item])

        cardinality_constraint[item] = LConstraint(lhs, "==", rhs)

    l_constraint(model, "cardinality_constraint", cardinality_constraint, partitions)

    objective = LExpression([(1, model.y[w]) for w in model.W])
    l_objective(model, objective, sense=maximize)


    if write_lp:
        model.write(filename=join(output_folder, 'model.lp'),
                    format=ProblemFormat.cpxlp,
                    io_options={'symbolic_solver_labels': True})

    return model, indices















# def build_model_relaxation(model_parameters, input_data, formulation,
#                            subgradient_method=None, y_dual=None, y_keep=None, multiplier=None,
#                            output_folder=None, write_lp=False):
#
#     D = input_data['criticality_data']
#     no_locations = D.shape[1]
#     no_windows = D.shape[0]
#
#     partitions = [item for item in model_parameters['deployment_vector']]
#     n = sum(dict_deployment[item] for item in dict_deployment)
#
#     indices = input_data['location_indices']
#     beta = input_data['geographical_coverage']
#
#     model = ConcreteModel()
#
#     model.W = arange(1, no_windows + 1)
#     model.L = arange(1, no_locations + 1)
#     model.partitions = arange(1, k + 1)
#
#     model.c = int(floor(sum(n) * round((1 - beta), 2)) + 1)
#
#     if formulation == 'PartialConvex':
#
#         model.x = Var(model.L, within=Binary)
#         model.y = Var(model.W, within=NonNegativeReals, bounds=(0,1))
#
#         activation_constraint = {}
#
#         for w in model.W:
#             lhs = LExpression([(D[w - 1, l - 1], model.x[l]) for l in model.L])
#             rhs = LExpression([(model.c, model.y[w])])
#
#             activation_constraint[w] = LConstraint(lhs, ">=", rhs)
#
#         l_constraint(model, "activation_constraint", activation_constraint, list(model.W))
#
#         cardinality_constraint = {}
#
#         for i in model.partitions:
#             lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
#             rhs = LExpression(constant=n[i - 1])
#
#             cardinality_constraint[i] = LConstraint(lhs, "==", rhs)
#
#         l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))
#
#         objective = LExpression([(1, model.y[w]) for w in model.W])
#         l_objective(model, objective, sense=maximize)
#
#
#     elif formulation == 'Lagrangian':
#
#         model.x = Var(model.L, within=Binary)
#
#         if subgradient_method == 'Inexact':
#             model.y = Var(model.W, within=NonNegativeReals, bounds=(0, 1))
#         elif subgradient_method == 'Exact':
#             model.y = Var(model.W, within=Binary)
#         else:
#             raise ValueError(' This case is not available.')
#
#         activation_constraint = {}
#
#         for w in y_keep:
#
#             lhs = LExpression([(model.D[w - 1, l - 1], model.x[l]) for l in model.L])
#             rhs = LExpression([(model.c, model.y[w])])
#
#             activation_constraint[w] = LConstraint(lhs, ">=", rhs)
#
#         l_constraint(model, "activation_constraint", activation_constraint, y_keep)
#
#         cardinality_constraint = {}
#
#         for i in model.partitions:
#             lhs = LExpression([(1, model.x[l]) for l in indices[i - 1]])
#             rhs = LExpression(constant=n[i - 1])
#
#             cardinality_constraint[i] = LConstraint(lhs, "==", rhs)
#
#         l_constraint(model, "cardinality_constraint", cardinality_constraint, list(model.partitions))
#
#         lc = dict(zip(y_dual, multiply(model.c, array(list(multiplier.values())))))
#         dx = dot(array(list(multiplier.values())),
#                  delete(model.D, [i - 1 for i in y_keep], axis=0))
#
#         objective = LExpression()
#         objective.variables.extend([(1, model.y[w]) for w in model.W])
#         objective.variables.extend([(dx[l - 1], model.x[l]) for l in model.L])
#         objective.variables.extend([(-lc[w], model.y[w]) for w in y_dual])
#
#         model.objective = Objective(expr=0., sense=maximize)
#         model.objective._expr = _build_sum_expression(objective.variables)
#
#
#     else:
#         raise ValueError(' This optimization setup is not available yet. Retry.')
#
#
#     if write_lp:
#         model.write(filename=join(output_folder, 'model.lp'),
#                     format=ProblemFormat.cpxlp,
#                     io_options={'symbolic_solver_labels': True})
#
#     return model