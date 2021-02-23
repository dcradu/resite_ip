from os.path import join

from numpy import arange
from pyomo.environ import ConcreteModel, Var, Binary, maximize
from pyomo.opt import ProblemFormat
from pypsa.opt import l_constraint, LConstraint, l_objective, LExpression

from src.helpers import custom_log, xarray_to_ndarray
from src.tools import read_database, return_filtered_coordinates, selected_data, return_output, \
    resource_quality_mapping, critical_window_mapping, retrieve_index_dict, critical_data_position_mapping


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

    regions = model_parameters['regions']
    technologies = model_parameters['technologies']

    time_horizon = model_parameters['time_slice']
    spatial_resolution = model_parameters['spatial_resolution']

    path_resource_data = model_parameters['path_resource_data'] + '/' + str(spatial_resolution)
    path_transfer_function_data = model_parameters['path_transfer_function_data']
    path_population_density_data = model_parameters['path_population_density_data']
    path_land_data = model_parameters['path_land_data']
    path_load_data = model_parameters['path_load_data']
    path_shapefile_data = model_parameters['path_shapefile_data']

    resource_quality_layer = model_parameters['resource_quality_layer']
    population_layer = model_parameters['population_density_layer']
    protected_areas_layer = model_parameters['protected_areas_layer']
    bathymetry_layer = model_parameters['bathymetry_layer']
    forestry_layer = model_parameters['forestry_layer']
    orography_layer = model_parameters['orography_layer']
    water_mask_layer = model_parameters['water_mask_layer']
    latitude_layer = model_parameters['latitude_layer']
    distance_layer = model_parameters['distance_layer']

    delta = model_parameters['delta']
    alpha = model_parameters['alpha']
    measure = model_parameters['smooth_measure']
    norm_type = model_parameters['norm_type']

    database = read_database(path_resource_data)
    filtered_coordinates = return_filtered_coordinates(database, spatial_resolution, technologies, regions,
                                                     path_land_data, path_resource_data,
                                                     path_shapefile_data, path_population_density_data,
                                                     resource_quality_layer=resource_quality_layer,
                                                     population_density_layer=population_layer,
                                                     protected_areas_layer=protected_areas_layer,
                                                     orography_layer=orography_layer,
                                                     forestry_layer=forestry_layer,
                                                     water_mask_layer=water_mask_layer,
                                                     bathymetry_layer=bathymetry_layer,
                                                     latitude_layer=latitude_layer,
                                                     distance_layer=distance_layer)

    truncated_data = selected_data(database, filtered_coordinates, time_horizon)
    output_data = return_output(truncated_data, path_transfer_function_data)

    smooth_data = resource_quality_mapping(output_data, delta, measure)
    critical_data = critical_window_mapping(smooth_data, alpha, delta, regions, time_horizon, path_load_data, norm_type)
    position_mapping = critical_data_position_mapping(critical_data)

    output_dict = {
                'coordinates_data': filtered_coordinates,
                'capacity_factor_data': output_data,
                'criticality_data': xarray_to_ndarray(critical_data),
                'site_positions_in_matrix': position_mapping}

    custom_log(' Input data read...')

    return output_dict


def build_model(model_parameters, coordinate_dict, D, output_folder, write_lp=False):
    """Model build-up.

    Parameters:

    ------------


    Returns:

    -----------

    """

    no_windows = D.shape[0]
    no_locations = D.shape[1]

    n, dict_deployment, partitions, indices = retrieve_index_dict(model_parameters, coordinate_dict)

    c = model_parameters['solution_method']['BB']['c']
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
