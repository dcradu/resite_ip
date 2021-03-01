import pickle
from ast import literal_eval
from copy import deepcopy
from glob import glob
from os import listdir
from os.path import join, isfile

import dask.array as da
import geopy.distance
import xarray as xr
import xarray.ufuncs as xu
from geopandas import read_file
from numpy import arange, interp, float32, datetime64, sqrt, asarray, newaxis, sum, max, unique, \
    radians, cos, sin, arctan2, zeros
from pandas import read_csv, Series, DataFrame, date_range, concat, MultiIndex, to_datetime
from shapely.geometry import Point
from shapely.ops import nearest_points
from windpowerlib import power_curves, wind_speed

from helpers import filter_onshore_offshore_locations, union_regions, return_coordinates_from_shapefiles, \
    concatenate_dict_keys, return_dict_keys, chunk_split, collapse_dict_region_level, read_inputs, \
    retrieve_load_data_partitions, get_partition_index, return_region_divisions


def read_database(data_path, spatial_resolution):
    """
    Reads resource database from .nc files.

    Parameters
    ----------
    data_path : str
    spatial_resolution: float

    Returns
    -------
    dataset: xarray.Dataset

    """
    file_path = join(data_path, 'input/resource_data', str(spatial_resolution))
    # Read through all files, extract the first 2 characters (giving the
    # macro-region) and append in a list that will keep the unique elements.
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    areas = []
    datasets = []
    for item in files:
        areas.append(item[:2])
    areas_unique = list(set(areas))

    # For each available area use the open_mfdataset method to open
    # all associated datasets, while directly concatenating on time dimension
    # and also aggregating (longitude, latitude) into one single 'location'. As
    # well, data is read as float32 (memory concerns).
    for area in areas_unique:
        file_list = [file for file in glob(file_path + '/*.nc') if area in file]
        ds = xr.open_mfdataset(file_list,
                               combine='by_coords',
                               chunks={'latitude': 100, 'longitude': 100}).stack(locations=('longitude', 'latitude'))
        datasets.append(ds.astype(float32))

    # Concatenate all regions on locations.
    dataset = xr.concat(datasets, dim='locations')
    # Removing duplicates potentially there from previous concat of multiple regions.
    _, index = unique(dataset['locations'], return_index=True)
    dataset = dataset.isel(locations=index)
    # Sorting dataset on coordinates (mainly due to xarray peculiarities between concat and merge).
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    # Remove attributes from datasets. No particular use, looks cleaner.
    dataset.attrs = {}

    return dataset


def filter_locations_by_layer(regions, start_coordinates, model_params, tech_params, which=None):
    """
    Filters (removes) locations from the initial set following various
    land-, resource-, population-based criteria.

    Parameters
    ----------
    regions : list
        Region list.
    start_coordinates : list
        List of initial (starting) coordinates.
    model_params : dict

    tech_params : dict

    which : str
        Filter to be applied.

    Returns
    -------
    coords_to_remove : list
        List of coordinates to be removed from the initial set.

    """

    coords_to_remove = None
    data_path = model_params['data_path']

    if which == 'protected_areas':

        protected_areas_selection = tech_params['protected_areas_selection']
        threshold_distance = tech_params['protected_areas_distance_threshold']
        coords_to_remove = []

        areas_fn = join(data_path, 'input/land_data', 'WDPA_Feb2019-shapefile-points.shp')
        dataset = read_file(areas_fn)

        lons = []
        lats = []

        # Retrieve the geopandas Point objects and their coordinates
        for item in protected_areas_selection:
            for index, row in dataset.iterrows():
                if row['IUCN_CAT'] == item:
                    lons.append(round(row.geometry[0].x, 2))
                    lats.append(round(row.geometry[0].y, 2))

        protected_coords = list(zip(lons, lats))

        # Compute distance between reference coordinates and Points
        for i in start_coordinates:
            for j in protected_coords:
                lat1 = radians(i[1])
                lon1 = radians(i[0])
                lat2 = radians(j[1])
                lon2 = radians(j[0])

                dlon = lon2 - lon1
                dlat = lat2 - lat1

                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * arctan2(sqrt(a), sqrt(1 - a))

                distance = 6371. * c

                if distance < threshold_distance:
                    coords_to_remove.append(i)

    elif which == 'resource_quality':

        database = read_database(data_path, model_params['spatial_resolution'])

        if tech_params['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 +
                                     database.v100 ** 2)
        elif tech_params['resource'] == 'solar':
            array_resource = database.ssrd / 3600.
        else:
            raise ValueError (" This resource is not available.")

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < tech_params['resource_threshold'])
        coords_mask_resource = mask_resource[mask_resource.notnull()].locations.values.tolist()
        coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_resource]
        coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

    elif which == 'latitude':

        latitude_threshold = tech_params['latitude_threshold']
        coords_mask_latitude = [(lon, lat) for (lon, lat) in start_coordinates if lat > latitude_threshold]
        coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_latitude]
        coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

    elif which == 'distance':

        distance_threshold_min = tech_params['distance_threshold_min']
        distance_threshold_max = tech_params['distance_threshold_max']

        offshore_points = filter_onshore_offshore_locations(start_coordinates,
                                                            model_params['data_path'],
                                                            model_params['spatial_resolution'],
                                                            tech_params,
                                                            tech='wind_offshore')
        onshore_shape = union_regions(regions, data_path, which='onshore', prepped=False)

        offshore_distances = {key: None for key in offshore_points}
        for key in offshore_distances:
            p = Point(key)
            closest_point_geom = nearest_points(p, onshore_shape)[1]
            closest_point = (closest_point_geom.x, closest_point_geom.y)
            offshore_distances[key] = geopy.distance.geodesic(key, closest_point).km

        offshore_locations = {k: v for k, v in offshore_distances.items() if ((v < distance_threshold_min) |
                                                                              (v >= distance_threshold_max))}
        coords_mask_distance = list(offshore_locations.keys())
        coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_distance]
        coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

    elif which == 'orography':

        orography_fn = 'ERA5_orography_characteristics_20181231_' + str(model_params['spatial_resolution']) + '.nc'
        orography_path = join(data_path, 'input/land_data', orography_fn)
        dataset = xr.open_dataset(orography_path).astype(float32)
        dataset = dataset.sortby([dataset.longitude, dataset.latitude])
        dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                     + 180) % 360) - 180)).sortby('longitude')
        dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))

        altitude_threshold = tech_params['altitude_threshold']
        array_altitude = dataset['z'] / 9.80665

        slope_threshold = tech_params['terrain_slope_threshold']
        array_slope = dataset['slor']

        mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
        mask_slope = array_slope.where(array_slope.data > slope_threshold)

        coords_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()
        coords_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

        coords_mask_orography = list(set(coords_mask_altitude).union(set(coords_mask_slope)))
        coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_orography]
        coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

    elif which in ['forestry', 'water_mask', 'bathymetry']:

        surface_fn = 'ERA5_surface_characteristics_20181231_' + str(model_params['spatial_resolution']) + '.nc'
        surface_path = join(data_path, 'input/land_data', surface_fn)
        dataset = xr.open_dataset(surface_path).astype(float32)
        dataset = dataset.sortby([dataset.longitude, dataset.latitude])
        dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                     + 180) % 360) - 180)).sortby('longitude')
        dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))

        if which == 'forestry':

            forestry_threshold = tech_params['forestry_ratio_threshold']
            array_forestry = dataset['cvh']

            mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
            coords_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()
            coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_forestry]
            coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

        elif which == 'water_mask':

            watermask_threshold = 0.4
            array_watermask = dataset['lsm']

            mask_watermask = array_watermask.where(array_watermask.data < watermask_threshold)
            coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()
            coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_watermask]
            coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

        elif which == 'bathymetry':

            depth_threshold_low = tech_params['depth_threshold_low']
            depth_threshold_high = tech_params['depth_threshold_high']

            array_watermask = dataset['lsm']
            # max depth is 999.
            array_bathymetry = dataset['wmb'].fillna(0.)

            mask_offshore = array_bathymetry.where(((array_bathymetry.data < depth_threshold_low) |
                                                    (array_bathymetry.data > depth_threshold_high)) |
                                                    (array_watermask.data > 0.1))
            coords_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()
            coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_offshore]
            coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

    elif which == 'population_density':

        population_fn = 'gpw_v4_population_density_adjusted_rev11_0.5.nc'
        population_path = join(data_path, 'input/population_density', population_fn)
        dataset = xr.open_dataset(population_path)

        varname = [item for item in dataset.data_vars][0]
        dataset = dataset.rename({varname: 'data'})
        # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
        data_pop = dataset.sel(raster=5)

        array_pop_density = \
            data_pop['data'].interp(longitude=sorted(list(set([item[0] for item in start_coordinates]))),
                                    latitude=sorted(list(set([item[1] for item in start_coordinates])))[::-1],
                                    method='nearest').fillna(0.)
        # Temporary, to reduce the size of this ds, which is anyway read in each iteration.
        min_lon, max_lon, min_lat, max_lat = -11., 32., 35., 80.
        mask_lon = (array_pop_density.longitude >= min_lon) & (array_pop_density.longitude <= max_lon)
        mask_lat = (array_pop_density.latitude >= min_lat) & (array_pop_density.latitude <= max_lat)

        pop_ds = array_pop_density.where(mask_lon & mask_lat, drop=True).stack(locations=('longitude', 'latitude'))

        population_density_threshold_low = tech_params['population_density_threshold_low']
        population_density_threshold_high = tech_params['population_density_threshold_high']

        mask_population = pop_ds.where((pop_ds.data < population_density_threshold_low) |
                                       (pop_ds.data > population_density_threshold_high))
        coords_mask_population = mask_population[mask_population.notnull()].locations.values.tolist()
        coords_mask = [(round(lon, 2), round(lat, 2)) for (lon, lat) in coords_mask_population]
        coords_to_remove = sorted(list(set(start_coordinates).intersection(set(coords_mask))))

    else:
        raise ValueError(' Layer {} is not available.'.format(str(which)))

    return coords_to_remove


def return_filtered_coordinates(dataset, model_params, tech_params):
    """
    Returns the set of potential deployment locations for each region and available technology.

    Parameters
    ----------
    dataset: xr.Dataset
        Resource dataset.
    model_params: dict
        Model parameters.
    tech_params: dict
        Technology parameters.

    Returns
    -------
    output_dict : dict
        Dict object storing potential locations sets per region and technology.

    """
    technologies = model_params['technologies']
    regions = model_params['regions']
    output_dict = {region: {tech: None for tech in technologies} for region in regions}
    coordinates_dict = {key: None for key in technologies}

    region_shape = union_regions(regions, model_params['data_path'], which='both')
    coordinates = return_coordinates_from_shapefiles(dataset, region_shape)
    start_coordinates = {(lon, lat): None for (lon, lat) in coordinates}
    for (lon, lat) in start_coordinates:
        start_coordinates[(lon, lat)] = (round(lon, 2), round(lat, 2))

    for tech in technologies:
        coords_to_remove = []
        tech_dict = tech_params[tech]

        for layer in tech_dict['filters']:
            to_remove_from_filter = filter_locations_by_layer(regions, list(start_coordinates.values()),
                                                              model_params, tech_dict, which=layer)
            coords_to_remove.extend(to_remove_from_filter)

        original_coordinates_list = []
        start_coordinates_reversed = dict(map(reversed, start_coordinates.items()))
        for item in set(start_coordinates.values()).difference(set(coords_to_remove)):
            original_coordinates_list.append(start_coordinates_reversed[item])
        coordinates_dict[tech] = original_coordinates_list

        unique_list_of_points = []
        for region in regions:

            shape_region = union_regions([region], model_params['data_path'], which=tech_dict['where'])
            points_in_region = return_coordinates_from_shapefiles(dataset, shape_region)

            points_to_keep = list(set(coordinates_dict[tech]).intersection(set(points_in_region)))
            output_dict[region][tech] = [p for p in points_to_keep if p not in unique_list_of_points]
            unique_list_of_points.extend(points_to_keep)

            if len(output_dict[region][tech]) > 0:
                print(f"{len(output_dict[region][tech])} {tech} sites in {region}.")
            else:
                print(f"No {tech} sites in {region}.")

    for key, value in output_dict.items():
        output_dict[key] = {k: v for k, v in output_dict[key].items() if len(v) > 0}

    return output_dict


def selected_data(dataset, input_dict, time_slice):
    """
    Slices xarray.Dataset based on relevant i) time horizon and ii) location sets.

    Parameters
    ----------
    dataset : xarray.Dataset
        Complete resource dataset.
    input_dict : dict
        Dict object storing location sets per region and technology.
    time_slice : list
        List containing start and end timestamps for the study.

    Returns
    -------
    output_dict : dict
        Dict object storing sliced data per region and technology.

    """
    key_list = return_dict_keys(input_dict)

    output_dict = deepcopy(input_dict)

    datetime_start = datetime64(time_slice[0])
    datetime_end = datetime64(time_slice[1])

    # This is a test which raised some problems on Linux distros, where for some
    # unknown reason the dataset is not sorted on the time dimension.
    if (datetime_start < dataset.time.values[0]) or \
            (datetime_end > dataset.time.values[-1]):
        raise ValueError(' At least one of the time indices exceeds the available data.')

    for region, tech in key_list:

        dataset_temp = []

        for chunk in chunk_split(input_dict[region][tech], n=50):
            dataset_region = dataset.sel(locations=chunk,
                                         time=slice(datetime_start, datetime_end))
            dataset_temp.append(dataset_region)

        output_dict[region][tech] = xr.concat(dataset_temp, dim='locations')

    return output_dict


def return_output(input_dict, data_path, smooth_wind_power_curve=True):
    """
    Applies transfer function to raw resource data.

    Parameters
    ----------
    input_dict : dict
        Dict object storing raw resource data.
    data_path : str
        Relative path to transfer function data.
    smooth_wind_power_curve : boolean
        If "True", the transfer function of wind assets replicates the one of a wind farm,
        rather than one of a wind turbine.


    Returns
    -------
    output_dict : dict
        Dict object storing capacity factors per region and technology.

    """
    key_list = return_dict_keys(input_dict)

    output_dict = deepcopy(input_dict)
    tech_dict = read_inputs('../config_techs.yml')

    wind_data_path = join(data_path, 'input/transfer_functions', 'data_wind_turbines.csv')
    data_converter_wind = read_csv(wind_data_path, sep=';', index_col=0)
    solar_data_path = join(data_path, 'input/transfer_functions', 'data_solar_modules.csv')
    data_converter_solar = read_csv(solar_data_path, sep=';', index_col=0)

    for region, tech in key_list:

        resource = tech.split('_')[0]

        if resource == 'wind':

            ###

            wind_speed_reference_height = 100.
            roughness = input_dict[region][tech].fsr

            # Compute the resultant of the two wind components.
            wind = xu.sqrt(input_dict[region][tech].u100 ** 2 +
                           input_dict[region][tech].v100 ** 2)

            wind_mean = wind.mean(dim='time')
            # wind_mean.unstack('locations').plot(x='longitude', y='latitude')
            # plt.show()

            # Split according to the IEC 61400 WTG classes
            wind_classes = {'IV': [0., 6.5], 'III': [6.5, 8.], 'II': [8., 9.5], 'I': [9.5, 99.]}

            output_array_list = []

            for cls in wind_classes:

                filtered_wind_data = wind_mean.where((wind_mean.data >= wind_classes[cls][0]) &
                                                     (wind_mean.data < wind_classes[cls][1]), 0)
                coords_classes = filtered_wind_data[da.nonzero(filtered_wind_data)].locations.values.tolist()

                if len(coords_classes) > 0:

                    wind_filtered = wind.sel(locations=coords_classes)
                    roughness_filtered = roughness.sel(locations=coords_classes)
                    ti = wind_filtered.std(dim='time') / wind_filtered.mean(dim='time')

                    converter = tech_dict[tech]['converter_' + str(cls)]
                    power_curve_array = literal_eval(data_converter_wind.loc['Power curve', converter])

                    wind_speed_references = asarray([i[0] for i in power_curve_array])
                    capacity_factor_references = asarray([i[1] for i in power_curve_array])
                    capacity_factor_references_pu = capacity_factor_references / max(capacity_factor_references)

                    wind_log = wind_speed.logarithmic_profile(wind_filtered.values,
                                                              wind_speed_reference_height,
                                                              float(
                                                                  data_converter_wind.loc['Hub height [m]', converter]),
                                                              roughness_filtered.values)
                    wind_data = da.from_array(wind_log, chunks='auto', asarray=True)

                    coordinates = wind_filtered.coords
                    dimensions = wind_filtered.dims

                    if smooth_wind_power_curve:
                        # Windpowerlib function here:
                        # windpowerlib.readthedocs.io/en/latest/temp/windpowerlib.power_curves.smooth_power_curve.html
                        capacity_factor_farm = \
                            power_curves.smooth_power_curve(Series(wind_speed_references),
                                                            Series(capacity_factor_references_pu),
                                                            standard_deviation_method='turbulence_intensity',
                                                            turbulence_intensity=float(ti.mean().values))

                        power_output = da.map_blocks(interp, wind_data,
                                                     capacity_factor_farm['wind_speed'].values,
                                                     capacity_factor_farm['value'].values).compute()

                    else:

                        power_output = da.map_blocks(interp, wind_data,
                                                     wind_speed_references,
                                                     capacity_factor_references_pu).compute()

                    output_array_interm = xr.DataArray(power_output, coords=coordinates, dims=dimensions)
                    output_array_list.append(output_array_interm)

                else:

                    continue

            output_array = xr.concat(output_array_list, dim='locations')

        elif resource == 'pv':

            converter = tech_dict[tech]['converter']

            # Get irradiance in W from J
            irradiance = input_dict[region][tech].ssrd / 3600.
            # Get temperature in C from K
            temperature = input_dict[region][tech].t2m - 273.15

            coordinates = input_dict[region][tech].ssrd.coords
            dimensions = input_dict[region][tech].ssrd.dims

            # Homer equation here:
            # https://www.homerenergy.com/products/pro/docs/latest/how_homer_calculates_the_pv_array_power_output.html
            # https://enphase.com/sites/default/files/Enphase_PVWatts_Derate_Guide_ModSolar_06-2014.pdf
            power_output = (float(data_converter_solar.loc['f', converter]) *
                            (irradiance / float(data_converter_solar.loc['G_ref', converter])) *
                            (1. + float(data_converter_solar.loc['k_P [%/C]', converter]) / 100. *
                            (temperature - float(data_converter_solar.loc['t_ref', converter]))))

            output_array = xr.DataArray(power_output, coords=coordinates, dims=dimensions)

        else:
            raise ValueError(' The resource specified is not available yet.')

        output_array = output_array.where(output_array > 0.01, other=0.0)
        output_dict[region][tech] = output_array.reindex_like(input_dict[region][tech])

    return output_dict


##############################################################################
def resource_quality_mapping(input_dict, siting_params):

    delta = siting_params['delta']
    measure = siting_params['smooth_measure']

    key_list = return_dict_keys(input_dict)
    output_dict = deepcopy(input_dict)

    for region, tech in key_list:

        if measure == 'mean':
            # The method here applies the mean over a rolling
            # window of length delta, centers the label and finally
            # drops NaN values resulted.
            time_array = input_dict[region][tech].rolling(time=delta, center=True).mean().dropna('time')

        elif measure == 'median':
            # The method here returns the median over a rolling
            # window of length delta, centers the label and finally
            # drops NaN values resulted.
            time_array = input_dict[region][tech].rolling(time=delta, center=True).median().dropna('time')

        else:

            raise ValueError(' Measure {} is not available.'.format(str(measure)))

        # Renaming coordinate and updating its values to integers.
        time_array = time_array.rename({'time': 'windows'})
        window_dataarray = time_array.assign_coords(windows=arange(1, time_array.windows.values.shape[0] + 1))

        output_dict[region][tech] = window_dataarray

    return output_dict


def critical_window_mapping(input_dict, model_params):

    regions = model_params['regions']
    date_slice = model_params['time_slice']
    alpha = model_params['siting_params']['alpha']
    delta = model_params['siting_params']['delta']
    norm_type = model_params['siting_params']['norm_type']
    data_path = model_params['data_path']

    key_list = return_dict_keys(input_dict)
    output_dict = deepcopy(input_dict)

    if alpha == 'load_central':

        l_norm = retrieve_load_data_partitions(data_path, date_slice, alpha, delta, regions, norm_type)
        # Flip axes
        alpha_reference = l_norm[:, newaxis]

        for region, tech in key_list:
            critical_windows = (input_dict[region][tech] > alpha_reference).astype(int)
            output_dict[region][tech] = critical_windows

    elif alpha == 'load_partition':

        for region, tech in key_list:
            l_norm = retrieve_load_data_partitions(data_path, date_slice, alpha, delta, region, norm_type)
            # Flip axes.
            alpha_reference = l_norm[:, newaxis]

            # Select region of interest within the dict value with 'tech' key.
            critical_windows = (input_dict[region][tech] > alpha_reference).astype(int)
            output_dict[region][tech] = critical_windows

    else:
        raise ValueError('No such alpha rule. Retry.')

    return output_dict


def sites_position_mapping(input_dict):

    key_list = return_dict_keys(input_dict)
    locations_list = []
    for region, tech in key_list:
        locations_list.extend([(tech, loc) for loc in input_dict[region][tech].locations.values.flatten()])
    locations_dict = dict(zip(locations_list, list(arange(len(locations_list)))))

    return locations_dict

##########################################################

def spatiotemporal_criticality_mapping(data_array, c):
    # Checks global criticality (returns 0 if critical, 1 otherwise) and computes
    # the criticality index for a given region by dividing the sum on dimension
    # 'windows' to its length.

    temporal_noncriticality = data_array.sum(dim='locations')
    spatiotemporal_noncriticality = (temporal_noncriticality >= c).astype(int).sum(dim='windows')

    return spatiotemporal_noncriticality


def retrieve_location_dict(x_values, model_parameters, site_positions):

    output_dict = {key: [] for key in model_parameters['technologies']}
    reversed_site_positions = dict(map(reversed, site_positions.items()))

    for tech in output_dict:
        for item, val in enumerate(x_values):
            if (val == 1.0) and (reversed_site_positions[item][0] == tech):
                output_dict[tech].append(reversed_site_positions[item][1])

    return output_dict


def retrieve_index_dict(deployment_vector, coordinate_dict):

    dict_deployment = concatenate_dict_keys(deployment_vector)
    n = sum(dict_deployment[item] for item in dict_deployment)
    partitions = [item for item in dict_deployment]
    indices = concatenate_dict_keys(get_partition_index(coordinate_dict))

    return n, dict_deployment, partitions, indices


def retrieve_site_data(model_parameters, deployment_dict, coordinates_dict, output_data, criticality_data,
                       location_mapping, c, site_coordinates, objective, output_folder, benchmarks=True):

    output_by_tech = collapse_dict_region_level(output_data)
    time_slice = model_parameters['time_slice']
    time_dt = date_range(start=time_slice[0], end=time_slice[1], freq='H')

    for tech in output_by_tech:
        _, index = unique(output_by_tech[tech].locations, return_index=True)
        output_by_tech[tech] = output_by_tech[tech].isel(locations=index)

    # Init coordinate set.
    tech_dict = {key: [] for key in list(output_by_tech.keys())}
    for tech in tech_dict:
        for region in coordinates_dict.keys():
            for t in coordinates_dict[region].keys():
                if t == tech:
                    tech_dict[tech].extend(sorted(coordinates_dict[region][t], key=lambda x: (x[0], x[1])))

    pickle.dump(tech_dict, open(join(output_folder, 'init_coordinates_dict.p'), 'wb'))

    # Retrieve complementary sites
    comp_site_data = {key: None for key in site_coordinates}
    for item in site_coordinates:
        comp_site_data[item] = {key: None for key in site_coordinates[item]}

    for tech in comp_site_data.keys():
        for coord in comp_site_data[tech].keys():
            comp_site_data[tech][coord] = output_by_tech[tech].sel(locations=coord).values.flatten()

    reform = {(outerKey, innerKey): values for outerKey, innerDict in comp_site_data.items() for innerKey, values in
              innerDict.items()}
    comp_site_data_df = DataFrame(reform, index=time_dt)
    pickle.dump(comp_site_data_df, open(join(output_folder,  'comp_site_data.p'), 'wb'))
    # Similarly done for any other deployment scheme done via the Julia heuristics.

    with open(join(output_folder, 'objective_comp.txt'), "w") as file:
        print(objective, file=file)

    if benchmarks:

        # Retrieve max sites
        key_list = return_dict_keys(output_data)
        output_location = deepcopy(output_data)

        for region, tech in key_list:
            n = deployment_dict[region][tech]
            output_data_sum = output_data[region][tech].sum(dim='time')
            if n != 0:
                locs = output_data_sum.argsort()[-n:].values
                output_location[region][tech] = sorted(output_data_sum.isel(locations=locs).locations.values.flatten())
            else:
                output_location[region][tech] = None

        output_location_per_tech = {key: [] for key in model_parameters['technologies']}
        for region in output_location:
            for tech in output_location[region]:
                if output_location[region][tech] is not None:
                    for item in output_location[region][tech]:
                        output_location_per_tech[tech].append(item)

        max_site_data = {key: None for key in model_parameters['technologies']}
        for item in model_parameters['technologies']:
            max_site_data[item] = {key: None for key in output_location_per_tech[item]}

        for tech in max_site_data.keys():
            for coord in max_site_data[tech].keys():
                max_site_data[tech][coord] = output_by_tech[tech].sel(locations=coord).values.flatten()

        reform = {(outerKey, innerKey): values for outerKey, innerDict in max_site_data.items() for innerKey, values in
                  sorted(innerDict.items())}
        max_site_data_df = DataFrame(reform, index=time_dt)
        pickle.dump(max_site_data_df, open(join(output_folder, 'prod_site_data.p'), 'wb'))

        objective_prod = get_objective_from_mapfile(max_site_data_df, location_mapping, criticality_data, c)
        with open(join(output_folder, 'objective_prod.txt'), "w") as file:
            print(objective_prod, file=file)

        # Capacity credit sites.

        load_data_fn = join(model_parameters['data_path'], 'input/load_data', 'load_2009_2018.csv')
        load_data = read_csv(load_data_fn, index_col=0)
        load_data.index = to_datetime(load_data.index)
        load_data = load_data[(load_data.index > time_slice[0]) & (load_data.index < time_slice[1])]

        df_list = []
        no_index = 0.05 * len(load_data.index)
        for country in deployment_dict:

            if country in load_data.columns:
                load_country = load_data.loc[:, country]
            else:
                countries = return_region_divisions([country], model_parameters['data_path'])
                load_country = load_data.loc[:, countries].sum(axis=1)
            load_country_max = load_country.nlargest(int(no_index)).index

            for tech in model_parameters['technologies']:

                country_data = output_data[country][tech]
                country_data_avg_at_load_max = country_data.sel(time=load_country_max).mean(dim='time')
                locs = country_data_avg_at_load_max.argsort()[-deployment_dict[country][tech]:].values
                country_sites = country_data_avg_at_load_max.isel(locations=locs).locations.values.flatten()

                xarray_data = country_data.sel(locations=country_sites)
                df_data = xarray_data.to_pandas()
                col_list_updated = [(tech, item) for item in df_data.columns]
                df_data.columns = MultiIndex.from_tuples(col_list_updated)
                df_list.append(df_data)

        capv_site_data_df = concat(df_list, axis=1)
        pickle.dump(capv_site_data_df, open(join(output_folder, 'capv_site_data.p'), 'wb'))

        objective_capv = get_objective_from_mapfile(capv_site_data_df, location_mapping, criticality_data, c)
        with open(join(output_folder, 'objective_capv.txt'), "w") as file:
            print(objective_capv, file=file)


def get_objective_from_mapfile(df_sites, mapping_file, criticality_data, c):

    sites = [item for item in df_sites.columns]
    positions_in_matrix = [mapping_file[s] for s in sites]

    xs = zeros(shape=criticality_data.shape[1])
    xs[positions_in_matrix] = 1

    return (criticality_data.dot(xs) >= c).astype(int).sum()
