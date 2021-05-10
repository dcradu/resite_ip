import pickle
from ast import literal_eval
from copy import deepcopy
from glob import glob
from os import listdir
from os.path import join, isfile

import dask
import dask.array as da
dask.config.set({"array.slicing.split_large_chunks": True})
import geopy.distance
import xarray as xr
import xarray.ufuncs as xu
from geopandas import read_file
from numpy import arange, interp, float32, datetime64, sqrt, asarray, newaxis, sum, max, unique, \
    radians, cos, sin, arctan2, zeros, ceil
from pandas import read_csv, Series, DataFrame, date_range, concat, MultiIndex, to_datetime
from shapely.geometry import Point
from shapely.ops import nearest_points
from windpowerlib import power_curves, wind_speed

from helpers import filter_onshore_offshore_locations, union_regions, return_coordinates_from_shapefiles, \
    concatenate_dict_keys, return_dict_keys, chunk_split, collapse_dict_region_level, read_inputs, \
    smooth_load_data, get_partition_index, return_region_divisions, norm_load_by_deployments, norm_load_by_load

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
# logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)


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

    assert which in ['protected_areas', 'resource_quality', 'latitude', 'distance', 'orography',
                     'forestry', 'water_mask', 'bathymetry', 'population_density', 'legacy'], \
        f"Filtering layer {which} not available."

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

        assert tech_params['resource'] in ['wind', 'solar'], f"Resource {tech_params['resource']} not available."

        if tech_params['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 +
                                     database.v100 ** 2)
        elif tech_params['resource'] == 'solar':
            array_resource = database.ssrd / 3600.

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

    elif which == 'legacy':

        legacy_fn = 'aggregated_capacity.csv'
        legacy_path = join(data_path, 'input/legacy_data', legacy_fn)
        legacy_data = read_csv(legacy_path)

        legacy_data = legacy_data[(legacy_data['Plant'] == tech_params['resource'].capitalize()) &
                                  (legacy_data['Type'] == tech_params['deployment'].capitalize()) &
                                  (legacy_data['Capacity (GW)'] >= tech_params['legacy_min'])]

        coords_to_remove = list(zip(legacy_data.Longitude, legacy_data.Latitude))

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
    legacy_dict = {key: None for key in technologies}

    region_shape = union_regions(regions, model_params['data_path'], which='both')
    coordinates = return_coordinates_from_shapefiles(dataset, region_shape)
    start_coordinates = {(lon, lat): None for (lon, lat) in coordinates}
    for (lon, lat) in start_coordinates:
        start_coordinates[(lon, lat)] = (round(lon, 2), round(lat, 2))

    for tech in technologies:
        coords_to_remove = []
        coords_to_add = []
        tech_dict = tech_params[tech]

        for layer in tech_dict['filters']:
            to_remove_from_filter = filter_locations_by_layer(regions, list(start_coordinates.values()),
                                                              model_params, tech_dict, which=layer)
            if layer != 'legacy':
                coords_to_remove.extend(to_remove_from_filter)
            else:
                coords_to_add.extend(to_remove_from_filter)
        coords_to_remove = list(set(coords_to_remove).difference(set(coords_to_add)))
        legacy_dict[tech] = coords_to_add

        original_coordinates_list = []
        start_coordinates_reversed = {v: k for k, v in start_coordinates.items()}
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

    for key, value in output_dict.items():
        output_dict[key] = {k: v for k, v in output_dict[key].items() if len(v) > 0}

    return output_dict, legacy_dict


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
    assert (datetime_start >= dataset.time.values[0]) or (datetime_end <= dataset.time.values[-1]), \
        ' At least one of the time indices exceeds the available data.'

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

        assert resource in ['wind', 'solar'], f"Resource {resource} not available in the model."

        if resource == 'wind':

            ###

            wind_speed_reference_height = 100.
            roughness = input_dict[region][tech].fsr

            # Compute the resultant of the two wind components.
            wind = xu.sqrt(input_dict[region][tech].u100 ** 2 +
                           input_dict[region][tech].v100 ** 2)
            wind_mean = wind.mean(dim='time')

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

        output_array = output_array.where(output_array > 0.01, other=0.0)
        output_dict[region][tech] = output_array.reindex_like(input_dict[region][tech])

    return output_dict


##############################################################################
def resource_quality_mapping(input_dict, siting_params):

    delta = siting_params['delta']
    measure = siting_params['alpha']['smoothing']

    assert measure in ['mean', 'median'], f"Measure {measure} not available."

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

        # Renaming coordinate and updating its values to integers.
        time_array = time_array.rename({'time': 'windows'})
        window_dataarray = time_array.assign_coords(windows=arange(1, time_array.windows.values.shape[0] + 1))

        output_dict[region][tech] = window_dataarray

    return output_dict


def critical_window_mapping(time_windows_dict, potentials_dict, deployments_dict, model_params):

    regions = model_params['regions']
    date_slice = model_params['time_slice']
    sampling_rate = model_params['resampling_rate']
    alpha = model_params['siting_params']['alpha']
    delta = model_params['siting_params']['delta']
    data_path = model_params['data_path']
    load_coverage = model_params['load_coverage']

    key_list = return_dict_keys(time_windows_dict)
    output_dict = deepcopy(time_windows_dict)

    assert alpha['method'] in ['load', 'potential'], f"Criticality definition based on {alpha['method']} not available."
    assert alpha['coverage'] in ['partition', 'system'], f"Criticality coverage {alpha['coverage']} not available."
    assert alpha['norm'] in ['min', 'max'], f"Norm {alpha['norm']} not available."

    load_ds = smooth_load_data(data_path, regions, date_slice, delta, sampling_rate)

    if alpha['coverage'] == 'system':

        load_ds_system = load_ds.sum(axis=1)

        if alpha['method'] == 'potential':

            # Covering only a fraction of 30% of demand, as per EC expectations
            load_ds_system = load_ds_system.multiply(load_coverage)

            deployments = sum(deployments_dict[key][subkey] for key in deployments_dict
                              for subkey in deployments_dict[key])
            l_norm = norm_load_by_deployments(load_ds_system, deployments)
            # Flip axes
            l_norm = l_norm.values[:, newaxis]

            for region, tech in key_list:
                measure = time_windows_dict[region][tech] * potentials_dict[region][tech]
                output_dict[region][tech] = (measure > l_norm).astype(int)

        else:
            l_norm = norm_load_by_load(load_ds_system, alpha['norm'])
            # Flip axes
            l_norm = l_norm.values[:, newaxis]

            for region, tech in key_list:
                output_dict[region][tech] = (time_windows_dict[region][tech] > l_norm).astype(int)

    elif alpha['coverage'] == 'partition':

        for region, tech in key_list:

            load_ds_region = load_ds[region]

            if alpha['method'] == 'potential':
 
                # Covering only a fraction of the demand via offshore wind. EC suggests 30% EU-wide,
                # no data per country currently available
                load_ds_region = load_ds_region.multiply(load_coverage)
 
                deployments = sum(deployments_dict[key][subkey] for key in deployments_dict
                                  for subkey in deployments_dict[key] if key == region)
                l_norm = norm_load_by_deployments(load_ds_region, deployments)
                # Flipping axes.
                l_norm = l_norm.values[:, newaxis]

                measure = time_windows_dict[region][tech] * potentials_dict[region][tech]
                output_dict[region][tech] = (measure > l_norm).astype(int)

            else:
                l_norm = norm_load_by_load(load_ds_region, alpha['norm'])
                # Flipping axes.
                l_norm = l_norm.values[:, newaxis]

                output_dict[region][tech] = (time_windows_dict[region][tech] > l_norm).astype(int)

    return output_dict


def sites_position_mapping(input_dict):

    key_list = return_dict_keys(input_dict)
    locations_list = []
    for region, tech in key_list:
        locations_list.extend([(tech, loc) for loc in input_dict[region][tech].locations.values.flatten()])
    locations_dict = dict(zip(list(arange(len(locations_list))), locations_list))

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

    for tech in output_dict:
        for item, val in enumerate(x_values):
            if (val == 1.0) and (site_positions[item][0] == tech):
                output_dict[tech].append(site_positions[item][1])

    return output_dict


def retrieve_index_dict(deployment_vector, coordinate_dict):

    dict_deployment = concatenate_dict_keys(deployment_vector)
    n = sum(dict_deployment[item] for item in dict_deployment)
    partitions = [item for item in dict_deployment]
    indices = concatenate_dict_keys(get_partition_index(coordinate_dict))

    return n, dict_deployment, partitions, indices


def retrieve_site_data(model_parameters, capacity_factor_data, criticality_data, deployment_dict,
                       location_mapping, comp_site_coordinates, legacy_sites, output_folder, benchmark):

    sampling_rate = model_parameters['resampling_rate']
    c = int(ceil(model_parameters['siting_params']['c'] * sum(deployment_dict[r][t] for r in deployment_dict.keys()
                                                              for t in deployment_dict[r].keys())))

    output_by_tech = collapse_dict_region_level(capacity_factor_data)
    time_dt = date_range(start=model_parameters['time_slice'][0], end=model_parameters['time_slice'][1],
                         freq=f"{sampling_rate}H")

    for tech in output_by_tech:
        _, index = unique(output_by_tech[tech].locations, return_index=True)
        output_by_tech[tech] = output_by_tech[tech].isel(locations=index)

    # Retrieve complementary sites
    comp_site_data = {k1: {k2: None for k2 in comp_site_coordinates[k1]} for k1 in comp_site_coordinates}

    for tech in comp_site_data:
        for site in comp_site_data[tech]:
            comp_site_data[tech][site] = output_by_tech[tech].sel(locations=site).values.flatten()

    reform = {(outerKey, innerKey): values for outerKey, innerDict in comp_site_data.items() for innerKey, values in
              innerDict.items()}
    comp_site_data_df = DataFrame(reform, index=time_dt).sort_index(axis='columns', level=1)
    pickle.dump(comp_site_data_df, open(join(output_folder,  'comp_site_data.p'), 'wb'))

    objective_comp = get_objective_from_mapfile(comp_site_data_df, location_mapping, criticality_data, c)
    with open(join(output_folder, 'objective_comp.txt'), "w") as file:
        print(objective_comp, file=file)

    if benchmark == 'PROD':
        # Retrieve max sites
        key_list = return_dict_keys(capacity_factor_data)
        output_location = deepcopy(capacity_factor_data)

        for region, tech in key_list:
            n = deployment_dict[region][tech]
            output_data_sum = capacity_factor_data[region][tech].sum(dim='time')

            if n > 0:
                locs_legacy = list(set(legacy_sites[tech]).intersection(set(output_data_sum.locations.values.flatten())))
                if len(locs_legacy) > 0:
                    locs_new = set(output_data_sum.locations.values.flatten()).difference(locs_legacy)
                    output_data_sum_no_legacy = output_data_sum.sel(locations=list(locs_new))
                    n_new = n - len(locs_legacy)
                    if n_new > 0:
                        locs_new_best_idx = output_data_sum_no_legacy.argsort()[-n_new:].values.flatten()
                        locs_new_best = output_data_sum_no_legacy.isel(locations=locs_new_best_idx).locations.values.flatten()
                        locs = sorted(list(set(locs_legacy).union(set(locs_new_best))))
                    else:
                        locs = sorted(locs_legacy)
                else:
                    locs_idx = output_data_sum.argsort()[-n:].values.flatten()
                    locs = output_data_sum.isel(locations=locs_idx).locations.values.flatten()
                output_location[region][tech] = locs
            else:
                output_location[region][tech] = None

        output_location_per_tech = {key: [] for key in model_parameters['technologies']}
        for region, tech in key_list:
            if output_location[region][tech] is not None:
                for site in output_location[region][tech]:
                    output_location_per_tech[tech].append(site)

        max_site_data = \
            {k1: {k2: None for k2 in output_location_per_tech[k1]} for k1 in model_parameters['technologies']}

        for tech in max_site_data.keys():
            for site in max_site_data[tech].keys():
                max_site_data[tech][site] = output_by_tech[tech].sel(locations=site).values.flatten()

        reform = {(outerKey, innerKey): values for outerKey, innerDict in max_site_data.items() for innerKey, values in
                  innerDict.items()}
        max_site_data_df = DataFrame(reform, index=time_dt).sort_index(axis='columns', level=1)
        pickle.dump(max_site_data_df, open(join(output_folder, 'prod_site_data.p'), 'wb'))

        objective_prod = get_objective_from_mapfile(max_site_data_df, location_mapping, criticality_data, c)
        with open(join(output_folder, 'objective_prod.txt'), "w") as file:
            print(objective_prod, file=file)

    elif benchmark == 'CAPV':

        # Capacity credit sites.
        load_data_fn = join(model_parameters['data_path'], 'input/load_data', 'load_entsoe_2006_2020_full.csv')
        load_data = read_csv(load_data_fn, index_col=0)
        load_data.index = to_datetime(load_data.index)
        load_data = load_data[(load_data.index > model_parameters['time_slice'][0]) &
                              (load_data.index < model_parameters['time_slice'][1])]

        df_list = []
        no_index = 0.05 * len(load_data.index)
        for region in deployment_dict:

            if region in load_data.columns:
                load_country = load_data.loc[:, region]
            else:
                countries = return_region_divisions([region], model_parameters['data_path'])
                load_country = load_data.loc[:, countries].sum(axis=1)
            load_country_max = load_country.nlargest(int(no_index)).index

            for tech in model_parameters['technologies']:

                country_data = capacity_factor_data[region][tech]
                country_data_avg_at_load_max = country_data.sel(time=load_country_max).mean(dim='time')
                n = deployment_dict[region][tech]
                if n > 0:
                    locs_legacy = list(set(legacy_sites[tech]).intersection(set(country_data_avg_at_load_max.locations.values.flatten())))
                    if len(locs_legacy) > 0:
                        locs_new = set(country_data_avg_at_load_max.locations.values.flatten()).difference(locs_legacy)
                        country_data_no_legacy = country_data_avg_at_load_max.sel(locations=list(locs_new))
                        n_new = n - len(locs_legacy)
                        if n_new > 0:
                            locs_new_best_idx = country_data_no_legacy.argsort()[-n_new:].values.flatten()
                            locs_new_best = country_data_no_legacy.isel(locations=locs_new_best_idx).locations.values.flatten()
                            locs = sorted(list(set(locs_legacy).union(set(locs_new_best))))
                        else:
                            locs = sorted(locs_legacy)
                    else:
                        locs_idx = country_data_avg_at_load_max.argsort()[-n:].values.flatten()
                        locs = country_data_avg_at_load_max.isel(locations=locs_idx).locations.values.flatten()  
                    country_sites = locs
                else:
                    country_sites = None

                xarray_data = country_data.sel(locations=country_sites)
                df_data = xarray_data.to_pandas()
                col_list_updated = [(tech, item) for item in df_data.columns]
                df_data.columns = MultiIndex.from_tuples(col_list_updated)
                df_list.append(df_data)

        capv_site_data_df = concat(df_list, axis=1).sort_index(axis='columns', level=1)
        pickle.dump(capv_site_data_df, open(join(output_folder, 'capv_site_data.p'), 'wb'))

        objective_capv = get_objective_from_mapfile(capv_site_data_df, location_mapping, criticality_data, c)
        with open(join(output_folder, 'objective_capv.txt'), "w") as file:
            print(objective_capv, file=file)


def get_objective_from_mapfile(df_sites, mapping_file, criticality_data, c):

    sites = [item for item in df_sites.columns]
    mapping_file_reversed = {v: k for k, v in mapping_file.items()}
    positions_in_matrix = [mapping_file_reversed[s] for s in sites]

    xs = zeros(shape=criticality_data.shape[1])
    xs[positions_in_matrix] = 1

    return (criticality_data.dot(xs) >= c).astype(int).sum()
