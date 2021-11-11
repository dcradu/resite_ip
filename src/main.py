import pickle
import yaml
from os.path import join, isdir
from os import makedirs
from numpy import float64
from time import strftime
import julia

from helpers import read_inputs, custom_log, xarray_to_ndarray, \
                    generate_jl_input, get_deployment_vector
from tools import read_database, return_filtered_coordinates, \
                  selected_data, return_output, resource_quality_mapping, \
                  critical_window_mapping, sites_position_mapping

if __name__ == '__main__':

    model_parameters = read_inputs('../config_model.yml')
    siting_parameters = model_parameters['siting_params']
    tech_parameters = read_inputs('../config_techs.yml')

    data_path = model_parameters['data_path']
    julia_path = model_parameters['julia_models_path']
    spatial_resolution = model_parameters['spatial_resolution']
    time_horizon = model_parameters['time_slice']

    deployment_dict = get_deployment_vector(model_parameters['regions'],
                                            model_parameters['technologies'],
                                            model_parameters['deployments'])

    database = read_database(data_path, spatial_resolution)
    site_coordinates = return_filtered_coordinates(database, model_parameters, tech_parameters)
    truncated_data = selected_data(database, site_coordinates, time_horizon)
    capacity_factors_data = return_output(truncated_data, data_path)
    time_windows_data = resource_quality_mapping(capacity_factors_data, siting_parameters)
    D = xarray_to_ndarray(critical_window_mapping(time_windows_data, model_parameters))
    site_positions = sites_position_mapping(time_windows_data)

    output_dir = join(data_path, f"output/{strftime('%Y%m%d_%H%M%S')}/")
    if not isdir(output_dir):
        makedirs(output_dir)

    with open(join(output_dir, 'config_model.yaml'), 'w') as outfile:
        yaml.dump(model_parameters, outfile, default_flow_style=False, sort_keys=False)
    with open(join(output_dir, 'config_techs.yaml'), 'w') as outfile:
        yaml.dump(tech_parameters, outfile, default_flow_style=False, sort_keys=False)

    c = float64(siting_parameters['c'])
    D = D.astype(float64)

    jl_dict = generate_jl_input(deployment_dict, site_coordinates)
    j = julia.Julia(compiled_modules=False)
    from julia import Main
    Main.include(julia_path)

    custom_log(f" {siting_parameters['algorithm']} chosen to solve the instance.")

    if siting_parameters['algorithm'] == 'MIP':

        x, obj = Main.MIP(D, c, float64(jl_dict['k']))

    elif siting_parameters['algorithm'] == 'MIR':

        x, obj = Main.MIR(D, c, float64(jl_dict['k']))

    elif siting_parameters['algorithm'] == 'RG':

        # TODO: At this stage, this runs one single time.
        x, obj = Main.RG(D, c, float64(jl_dict['k']))

    elif siting_parameters['algorithm'] == 'RGP':

        p = siting_parameters['method_params']['RGP']['p']
        # TODO: At this stage, this runs one single time.
        x, obj = Main.RGP(D, c, float64(jl_dict['k']), p)

    elif siting_parameters['algorithm'] == 'RS':

        S = siting_parameters['method_params']['RS']['samples']
        x, obj = Main.RS(D, c, float64(jl_dict['k']), S)

    elif siting_parameters['algorithm'] == 'RSSA':

        I = siting_parameters['method_params']['RSSA']['no_iterations']
        N = siting_parameters['method_params']['RSSA']['no_epochs']
        r = siting_parameters['method_params']['RSSA']['radius']
        T_init = siting_parameters['method_params']['RSSA']['initial_temp']
        S = siting_parameters['method_params']['RSSA']['samples']
        x_init, _ = Main.RG(D, c, float64(jl_dict['k']), S)
        x, obj = Main.SA(D, c, float64(jl_dict['k']), x_init, I, N, r, T_init)

    elif siting_parameters['algorithm'] == 'MIRSA':

        I = siting_parameters['method_params']['MIRSA']['no_iterations']
        N = siting_parameters['method_params']['MIRSA']['no_epochs']
        r = siting_parameters['method_params']['MIRSA']['radius']
        T_init = siting_parameters['method_params']['MIRSA']['initial_temp']
        x, obj = Main.MIRSA(D, c, float64(jl_dict['k']), I, N, r, T_init)

    elif siting_parameters['algorithm'] == 'RGPSA':

        p = siting_parameters['method_params']['RGPSA']['p']
        n = siting_parameters['method_params']['RGPSA']['init_runs']
        I = siting_parameters['method_params']['RGPSA']['no_iterations']
        N = siting_parameters['method_params']['RGPSA']['no_epochs']
        r = siting_parameters['method_params']['RGPSA']['radius']
        T_init = siting_parameters['method_params']['RGPSA']['initial_temp']
        x, obj = Main.RGPSA(D, c, float64(jl_dict['k']), p, n, I, N, r, T_init)

    else:
        raise ValueError(f" Algorithm {siting_parameters['algorithm']} is not available.")

    pickle.dump(x, open(join(output_dir, 'solution_matrix.p'), 'wb'))
    pickle.dump(obj, open(join(output_dir, 'objective_vector.p'), 'wb'))
    custom_log(" Results written to disk.")
