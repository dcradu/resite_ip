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
from numpy import arange, interp, float32, datetime64, sqrt, floor, \
    asarray, newaxis, sum, \
    max, unique, radians, cos, sin, arctan2
from pandas import read_csv, Series, DataFrame, date_range
from shapely.geometry import Point
from shapely.ops import nearest_points
from windpowerlib import power_curves, wind_speed

from src.helpers import filter_onshore_offshore_locations, union_regions, return_coordinates_from_shapefiles_light, \
    return_region_divisions, read_legacy_capacity_data, retrieve_nodes_with_legacy_units, concatenate_dict_keys, \
    return_dict_keys, chunk_split, collapse_dict_region_level, dict_to_xarray, read_inputs, \
    retrieve_load_data_partitions, get_partition_index


def read_database(file_path):
    """
    Reads resource database from .nc files.

    Parameters
    ----------
    file_path : str
        Relative path to resource data.

    Returns
    -------
    dataset: xarray.Dataset

    """
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
                               chunks={'latitude': 20, 'longitude': 20}) \
            .stack(locations=('longitude', 'latitude')).astype(float32)
        datasets.append(ds)

    # Concatenate all regions on locations.
    dataset = xr.concat(datasets, dim='locations')
    # Removing duplicates potentially there from previous concat of multiple regions.
    _, index = unique(dataset['locations'], return_index=True)
    dataset = dataset.isel(locations=index)
    # dataset = dataset.sel(locations=~dataset.indexes['locations'].duplicated(keep='first'))
    # Sorting dataset on coordinates (mainly due to xarray peculiarities between concat and merge).
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    # Remove attributes from datasets. No particular use, looks cleaner.
    dataset.attrs = {}

    return dataset


def filter_locations_by_layer(tech_dict, regions,
                              start_coordinates, spatial_resolution,
                              path_land_data, path_resource_data, path_population_data, path_shapefile_data,
                              which='dummy', filename='dummy'):
    """
    Filters (removes) locations from the initial set following various
    land-, resource-, populatio-based criteria.

    Parameters
    ----------
    tech_dict : dict
        Dict object containing technical parameters and constraints of a given technology.
    start_coordinates : list
        List of initial (starting) coordinates.
    spatial_resolution : float
        Spatial resolution of the resource data.
    path_land_data : str
        Relative path to land data.
    path_resource_data : str
        Relative path to resource data.
    path_population_data : str
        Relative path to population density data.
    which : str
        Filter to be applied.
    filename : str
        Name of the file; associated with the filter type.

    Returns
    -------
    coords_to_remove : list
        List of coordinates to be removed from the initial set.

    """
    if which == 'protected_areas':

        protected_areas_selection = tech_dict['protected_areas_selection']
        threshold_distance = tech_dict['protected_areas_distance_threshold']

        coords_to_remove = []

        R = 6371.

        dataset = read_file(join(path_land_data, filename))

        lons = []
        lats = []

        # Retrieve the geopandas Point objects and their coordinates
        for item in protected_areas_selection:
            for index, row in dataset.iterrows():
                if (row['IUCN_CAT'] == item):
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

                distance = R * c

                if distance < threshold_distance:
                    coords_to_remove.append(i)

    elif which == 'resource':

        database = read_database(path_resource_data)

        if tech_dict['resource'] == 'wind':
            array_resource = xu.sqrt(database.u100 ** 2 +
                                     database.v100 ** 2)
        elif tech_dict['resource'] == 'solar':
            array_resource = database.ssrd / 3600.

        array_resource_mean = array_resource.mean(dim='time')
        mask_resource = array_resource_mean.where(array_resource_mean.data < tech_dict['resource_threshold'])
        coords_mask_resource = mask_resource[mask_resource.notnull()].locations.values.tolist()
        coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_resource)))

    elif which == 'latitude':

        latitude_threshold = tech_dict['latitude_threshold']

        coords_mask_latitude = [item for item in start_coordinates if item[1] > latitude_threshold]
        # print(coords_mask_latitude)
        coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_latitude)))

    elif which == 'distance':

        distance_threshold_min = tech_dict['distance_threshold_min']
        distance_threshold_max = tech_dict['distance_threshold_max']

        offshore_points = filter_onshore_offshore_locations(start_coordinates,
                                                            spatial_resolution,
                                                            'wind_offshore')
        onshore_shape = union_regions(regions, path_shapefile_data, which='onshore', prepped=False)

        offshore_distances = {key: None for key in offshore_points}
        for key in offshore_distances:
            p = Point(key)
            closest_point_geom = nearest_points(p, onshore_shape)[1]
            closest_point = (closest_point_geom.x, closest_point_geom.y)
            offshore_distances[key] = geopy.distance.geodesic(key, closest_point).km

        offshore_locations = {k: v for k, v in offshore_distances.items() if ((v < distance_threshold_min) |
                                                                              (v >= distance_threshold_max))}
        coords_mask_distance = list(offshore_locations.keys())
        coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_distance)))


    elif which in ['orography', 'forestry', 'water_mask', 'bathymetry']:

        dataset = xr.open_dataset(join(path_land_data, filename))
        dataset = dataset.sortby([dataset.longitude, dataset.latitude])

        dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                     + 180) % 360) - 180)).sortby('longitude')
        dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))

        if which == 'orography':

            altitude_threshold = tech_dict['altitude_threshold']
            slope_threshold = tech_dict['terrain_slope_threshold']

            array_altitude = dataset['z'] / 9.80665
            array_slope = dataset['slor']

            mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
            mask_slope = array_slope.where(array_slope.data > slope_threshold)

            coords_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()
            coords_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

            coords_mask_orography = list(set(coords_mask_altitude).union(set(coords_mask_slope)))
            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_orography)))

        elif which == 'forestry':

            forestry_threshold = tech_dict['forestry_ratio_threshold']

            array_forestry = dataset['cvh']

            mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
            coords_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_forestry)))

        elif which == 'water_mask':

            array_watermask = dataset['lsm']

            mask_watermask = array_watermask.where(array_watermask.data < 0.9)
            coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_watermask)))

        elif which == 'bathymetry':

            depth_threshold_low = tech_dict['depth_threshold_low']
            depth_threshold_high = tech_dict['depth_threshold_high']

            array_watermask = dataset['lsm']
            # Careful with this one because max depth is 999.
            array_bathymetry = dataset['wmb'].fillna(0.)

            mask_offshore = array_bathymetry.where((
                                                           (array_bathymetry.data < depth_threshold_low) | (
                                                               array_bathymetry.data > depth_threshold_high)) | \
                                                   (array_watermask.data > 0.1))
            coords_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

            coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_offshore)))

    elif which == 'population':

        dataset = xr.open_dataset(join(path_population_data, filename))

        varname = [item for item in dataset.data_vars][0]
        dataset = dataset.rename({varname: 'data'})
        # The value of 5 for "raster" fetches data for the latest estimate available in the dataset, that is, 2020.
        data_pop = dataset.sel(raster=5)

        array_pop_density = data_pop['data'].interp(longitude=arange(-180, 180, float(spatial_resolution)),
                                                    latitude=arange(-89, 91, float(spatial_resolution))[::-1],
                                                    method='nearest').fillna(0.)
        array_pop_density = array_pop_density.stack(locations=('longitude', 'latitude'))

        population_density_threshold_low = tech_dict['population_density_threshold_low']
        population_density_threshold_high = tech_dict['population_density_threshold_high']

        mask_population = array_pop_density.where((array_pop_density.data < population_density_threshold_low) |
                                                  (array_pop_density.data > population_density_threshold_high))
        coords_mask_population = mask_population[mask_population.notnull()].locations.values.tolist()

        coords_to_remove = list(set(start_coordinates).intersection(set(coords_mask_population)))

    else:

        raise ValueError(' Layer {} is not available.'.format(str(which)))

    return coords_to_remove


def return_filtered_coordinates(dataset, spatial_resolution, technologies, regions,
                                path_land_data, path_resource_data, path_legacy_data, path_shapefile_data,
                                path_population_data,
                                resource_quality_layer=True, population_density_layer=True, protected_areas_layer=False,
                                orography_layer=True, forestry_layer=True, water_mask_layer=True, bathymetry_layer=True,
                                latitude_layer=True,
                                legacy_layer=True, distance_layer=True):
    """
    Returns the set of potential deployment locations for each region and available technology.

    Parameters
    ----------
    coordinates_in_region : list
        List of coordinate pairs in the region of interest.
    spatial_resolution : float
        Spatial resolution of the resource data.
    technologies : list
        List of available technologies.
    regions : list
        List of regions.
    path_land_data : str

    path_resource_data : str

    path_legacy_data : str
        Relative path to existing capacities (for wind and solar PV) data.
    path_shapefile_data : str

    path_population_data : str

    resource_quality_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the average resource
        quality over the available time horizon. Resource quality threshold defined
        in the config_tech.yaml file.
    population_density_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the population density.
        Population density threshold defined in the config_tech.yaml file per each
        available technology.
    protected_areas_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the existance of protected
        areas in their vicinity. Distance threshold, as well as classes of areas are
         defined in the config_tech.yaml file.
    orography_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on their altitude and terrain
        slope. Both thresholds defined in the config_tech.yaml file for each individual
        technology.
    forestry_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on its forest cover share.
        Forest share threshold above which technologies are not built are defined
        in the config_tech.yaml file.
    water_mask_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it discards points in coordinates_in_region based on the water coverage share.
        Threshold defined in the config_tech.yaml file.
    bathymetry_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        (valid for offshore technologies) it discards points in coordinates_in_region
        based on the water depth. Associated threshold defined in the config_tech.yaml
        file for offshore and floating wind, respectively.
    legacy_layer : boolean
        "True" if the layer to be applied, "False" otherwise. If taken into account,
        it adds points to the final set based on the existence of RES projects in the area,
        thus avoiding a greenfield approach.

    Returns
    -------
    output_dict : dict
        Dict object storing potential locations sets per region and technology.

    """
    tech_config = read_inputs('../config_techs.yml')
    output_dict = {region: {tech: None for tech in technologies} for region in regions}
    final_coordinates = {key: None for key in technologies}

    region_shape = union_regions(regions, path_shapefile_data, which='both')

    for tech in technologies:

        tech_dict = tech_config[tech]
        # region_shapefile_data = return_region_shapefile(region, path_shapefile_data)
        start_coordinates = return_coordinates_from_shapefiles_light(dataset, region_shape)

        if resource_quality_layer:

            coords_to_remove_resource = \
                filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                          path_land_data, path_resource_data, path_population_data, path_shapefile_data,
                                          which='resource')
        else:
            coords_to_remove_resource = []

        if latitude_layer:

            coords_to_remove_latitude = \
                filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                          path_land_data, path_resource_data, path_population_data, path_shapefile_data,
                                          which='latitude')
        else:
            coords_to_remove_latitude = []

        if tech_dict['deployment'] in ['onshore', 'utility', 'residential']:

            if population_density_layer:
                filename = 'gpw_v4_population_density_rev11_' + str(spatial_resolution) + '.nc'
                coords_to_remove_population = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='population', filename=filename)

            else:
                coords_to_remove_population = []

            if protected_areas_layer:
                filename = 'WDPA_Feb2019-shapefile-points.shp'
                coords_to_remove_protected_areas = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='protected_areas', filename=filename)
            else:
                coords_to_remove_protected_areas = []

            if orography_layer:
                filename = 'ERA5_orography_characteristics_20181231_' + str(spatial_resolution) + '.nc'
                coords_to_remove_orography = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='orography', filename=filename)
            else:
                coords_to_remove_orography = []

            if forestry_layer:
                filename = 'ERA5_surface_characteristics_20181231_' + str(spatial_resolution) + '.nc'
                coords_to_remove_forestry = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='forestry', filename=filename)
            else:
                coords_to_remove_forestry = []

            if water_mask_layer:
                filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
                coords_to_remove_water = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='water_mask', filename=filename)
            else:
                coords_to_remove_water = []

            list_coords_to_remove = [coords_to_remove_resource,
                                     coords_to_remove_latitude,
                                     coords_to_remove_population,
                                     coords_to_remove_protected_areas,
                                     coords_to_remove_orography,
                                     coords_to_remove_forestry,
                                     coords_to_remove_water]
            coords_to_remove = set().union(*list_coords_to_remove)
            # Set difference between "global" coordinates and the sets computed in this function.
            updated_coordinates = set(start_coordinates) - coords_to_remove

        elif tech_dict['deployment'] in ['offshore', 'floating']:

            if bathymetry_layer:
                filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
                coords_to_remove_bathymetry = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='bathymetry', filename=filename)
            else:
                coords_to_remove_bathymetry = []

            if distance_layer:

                coords_to_remove_distance = \
                    filter_locations_by_layer(tech_dict, regions, start_coordinates, spatial_resolution,
                                              path_land_data, path_resource_data, path_population_data,
                                              path_shapefile_data,
                                              which='distance')

            else:

                coords_to_remove_distance = []

            list_coords_to_remove = [coords_to_remove_resource,
                                     coords_to_remove_bathymetry,
                                     coords_to_remove_distance]
            coords_to_remove = set().union(*list_coords_to_remove)
            updated_coordinates = set(start_coordinates) - coords_to_remove

        if legacy_layer:

            land_filtered_coordinates = filter_onshore_offshore_locations(start_coordinates,
                                                                          spatial_resolution, tech)
            legacy_dict = read_legacy_capacity_data(land_filtered_coordinates,
                                                    return_region_divisions(regions, path_shapefile_data),
                                                    tech, path_legacy_data)
            coords_to_add_legacy = retrieve_nodes_with_legacy_units(legacy_dict, regions, tech, path_shapefile_data)

            final_coordinates[tech] = set(updated_coordinates).union(set(coords_to_add_legacy))

        else:

            final_coordinates[tech] = updated_coordinates

        # if len(final_coordinates[tech]) > 0:
        #     from src.helpers import plot_basemap
        #     plot_basemap(final_coordinates[tech], title=tech)

    for region in regions:

        shape_region = union_regions([region], path_shapefile_data, which='both')
        points_in_region = return_coordinates_from_shapefiles_light(dataset, shape_region)

        for tech in technologies:

            points_to_take = list(set(final_coordinates[tech]).intersection(set(points_in_region)))

            output_dict[region][tech] = points_to_take

            if len(points_to_take) > 0:
                print(region, tech, len(points_to_take))
            else:
                print('{}, {} has no point.'.format(region, tech))

    for key, value in output_dict.items():
        output_dict[key] = {k: v for k, v in output_dict[key].items() if len(v) > 0}

    result_dict = {r: {t: [] for t in technologies} for r in regions}
    added_items = []
    for region, tech in return_dict_keys(output_dict):
        coords_region_tech = output_dict[region][tech]
        for item in coords_region_tech:
            if item in added_items:
                continue
            else:
                result_dict[region][tech].append(item)
                added_items.append(item)

    print('total no of points:', len(added_items))

    return result_dict


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


def return_output(input_dict, path_to_transfer_function, smooth_wind_power_curve=True):
    """
    Applies transfer function to raw resource data.

    Parameters
    ----------
    input_dict : dict
        Dict object storing raw resource data.
    path_to_transfer_function : str
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

    data_converter_wind = read_csv(join(path_to_transfer_function, 'data_wind_turbines.csv'), sep=';', index_col=0)
    data_converter_solar = read_csv(join(path_to_transfer_function, 'data_solar_modules.csv'), sep=';', index_col=0)

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
                        # https://windpowerlib.readthedocs.io/en/latest/temp/windpowerlib.power_curves.smooth_power_curve.html
                        capacity_factor_farm = power_curves.smooth_power_curve(Series(wind_speed_references),
                                                                               Series(capacity_factor_references_pu),
                                                                               standard_deviation_method='turbulence_intensity',
                                                                               turbulence_intensity=float(
                                                                                   ti.mean().values))

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

        elif resource == 'solar':

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
            power_output = (float(data_converter_solar.loc['f', converter]) * \
                            (irradiance / float(data_converter_solar.loc['G_ref', converter])) * \
                            (1. + float(data_converter_solar.loc['k_P [%/C]', converter]) / 100. * \
                             (temperature - float(data_converter_solar.loc['t_ref', converter]))))

            output_array = xr.DataArray(power_output, coords=coordinates, dims=dimensions)

        else:
            raise ValueError(' The resource specified is not available yet.')

        output_array = output_array.where(output_array > 0.01, other=0.0)

        output_dict[region][tech] = output_array

        # output_array.mean(dim='time').unstack('locations').plot(x='longitude', y='latitude')
        # plt.show()

    return output_dict


def resource_quality_mapping(input_dict, delta, measure):
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


def critical_window_mapping(input_dict,
                            alpha, delta,
                            regions, date_slice, path_load_data, norm_type):
    key_list = return_dict_keys(input_dict)

    output_dict = deepcopy(input_dict)

    if alpha == 'load_central':

        l_norm = retrieve_load_data_partitions(path_load_data, date_slice, alpha, delta, regions, norm_type)
        # Flip axes
        alpha_reference = l_norm[:, newaxis]

        for region, tech in key_list:
            critical_windows = (input_dict[region][tech] > alpha_reference).astype(int)
            output_dict[region][tech] = critical_windows

    elif alpha == 'load_partition':

        for region, tech in key_list:
            l_norm = retrieve_load_data_partitions(path_load_data, date_slice, alpha, delta, region, norm_type)
            # Flip axes.
            alpha_reference = l_norm[:, newaxis]

            # Select region of interest within the dict value with 'tech' key.
            critical_windows = (input_dict[region][tech] > alpha_reference).astype(int)
            output_dict[region][tech] = critical_windows

    else:
        raise ValueError('No such alpha rule. Retry.')

    return output_dict

def spatiotemporal_criticality_mapping(data_array, c, n):
    # Checks global criticality (returns 0 if critical, 1 otherwise) and computes
    # the criticality index for a given region by dividing the sum on dimension
    # 'windows' to its length.

    temporal_noncriticality = data_array.sum(dim='locations')
    spatiotemporal_noncriticality = (temporal_noncriticality >= c).astype(int).sum(dim='windows')

    return spatiotemporal_noncriticality


def retrieve_location_dict(instance, model_parameters, model_data, indices):
    output_dict = {key: [] for key in model_parameters['technologies']}

    coordinates = concatenate_dict_keys(model_data['coordinates_data'])

    for item in instance.x:
        if instance.x[item].value == 1.0:
            for key, index_list in indices.items():
                if item in index_list:
                    pos = [i for i, x in enumerate(index_list) if x == item][0]
                    output_dict[key[1]].append(coordinates[key][pos])

    return output_dict


def retrieve_location_dict_jl(jl_output, model_parameters, model_data, indices):

    output_dict = {key: [] for key in model_parameters['technologies']}

    coordinates = concatenate_dict_keys(model_data['coordinates_data'])
    for item, val in enumerate(jl_output, start=1):
        if val == 1.0:
            for key, index_list in indices.items():
                if item in index_list:
                    pos = [i for i, x in enumerate(index_list) if x == item][0]
                    output_dict[key[1]].append(coordinates[key][pos])

    return output_dict


def retrieve_index_dict(model_parameters, coordinate_dict):

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

    return n, dict_deployment, partitions, indices


def retrieve_site_data(model_parameters, model_data, output_folder, site_coordinates, suffix=None):

    deployment_dict = model_parameters['deployment_vector']
    output_by_tech = collapse_dict_region_level(model_data['capacity_factor_data'])
    time_slice = model_parameters['time_slice']
    time_dt = date_range(start=time_slice[0], end=time_slice[1], freq='H')

    for tech in output_by_tech:
        _, index = unique(output_by_tech[tech].locations, return_index=True)
        output_by_tech[tech] = output_by_tech[tech].isel(locations=index)

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

    if not suffix is None:
        name = 'comp_site_data'+str(suffix)+'.p'
    else:
        name = 'comp_site_data.p'
    pickle.dump(comp_site_data_df, open(join(output_folder, name), 'wb'))

    # Retrieve max sites
    output_data = model_data['capacity_factor_data']
    key_list = return_dict_keys(output_data)
    output_location = deepcopy(output_data)

    for region, tech in key_list:
        n = deployment_dict[region][tech]
        output_data_sum = output_data[region][tech].sum(dim='time')
        if n != 0:
            locs = output_data_sum.argsort()[-n:].values
            output_location[region][tech] = output_data_sum.isel(locations=locs).locations.values.flatten()
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
              innerDict.items()}
    max_site_data_df = DataFrame(reform, index=time_dt)

    pickle.dump(max_site_data_df, open(join(output_folder, 'max_site_data.p'), 'wb'))

    # Init coordinate set.
    coordinates_dict = model_data['coordinates_data']

    tech_dict = {key: [] for key in list(comp_site_data.keys())}
    for tech in tech_dict:
        for region in coordinates_dict.keys():
            for t in coordinates_dict[region].keys():
                if t == tech:
                    tech_dict[tech].extend(sorted(coordinates_dict[region][t], key=lambda x: (x[0], x[1])))

    pickle.dump(tech_dict, open(join(output_folder, 'init_coordinates_dict.p'), 'wb'))

    return output_location


def retrieve_max_run_criticality(max_sites, input_dict, parameters):
    capacity_factors_dict = input_dict['capacity_factor_data']
    alpha = parameters['alpha']
    delta = parameters['delta']
    c = parameters['c']
    measure = parameters['smooth_measure']
    regions = parameters['regions']
    time_horizon = parameters['time_slice']
    path_load_data = parameters['path_load_data']
    deployment_constraints = parameters['deployment_vector']
    norm_type = parameters['norm_type']

    depl = concatenate_dict_keys(deployment_constraints)
    n = sum(depl[item] for item in depl)

    key_list = return_dict_keys(max_sites)
    timeseries_dict = deepcopy(max_sites)

    for region, tech in key_list:

        if max_sites[region][tech] is not None:
            timeseries_dict[region][tech] = capacity_factors_dict[region][tech].sel(locations=max_sites[region][tech])

    for key, value in timeseries_dict.items():
        timeseries_dict[key] = {k: v for k, v in timeseries_dict[key].items() if v is not None}

    smooth_dict = resource_quality_mapping(timeseries_dict, delta, measure)
    critical_windows = critical_window_mapping(smooth_dict, alpha, delta, regions, time_horizon, path_load_data,
                                               norm_type)

    xarray_critical_windows = dict_to_xarray(critical_windows)
    no_windows = spatiotemporal_criticality_mapping(xarray_critical_windows, c, n)

    return no_windows.values