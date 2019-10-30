import sys
from numpy import arange, interp, load, float32, datetime64, where, sqrt, floor, \
                  asarray, ones, clip, array, newaxis, multiply, sum, tile, \
                  concatenate, delete, dot, vstack, take, full, exp, max, hstack, dtype, unique, pi
from numpy.random import uniform
from scipy import sin, cos, radians, arctan2
import xarray as xr
import xarray.ufuncs as xu
from glob import glob
import dask.array as da
from time import strftime
from os import makedirs, getcwd, listdir, remove
from os.path import isdir, abspath, join, isfile
from geopandas import read_file
import geopandas as gpd
from shapely import prepared
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import cascaded_union
from pandas import read_csv, DataFrame, date_range, MultiIndex
import yaml
from shutil import rmtree
from scipy.spatial import distance
from pyomo.environ import maximize, Objective
from pypsa.opt import LExpression, _build_sum_expression
from numpy import random as nr
from tqdm import tqdm
from joblib import Parallel, delayed
from random import choice, sample
from datetime import datetime
from operator import attrgetter
from itertools import takewhile





def list_chunks(l, n):
    """Splitting list in multiple chunks.

    Parameters:

    ------------

    l : list

    n : int


    Returns:

    ------------

    chunks : list

    """
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]





def filter_onshore_polys(polys, minarea=0.1, filterremote=True):
    """Filter onshore polygons based on i) polygon area and ii) distance to main land.

    Parameters:

    ------------

    polys : geopandas polygon
        Polygon or MultiPolygon associated to a given country.

    minarea : float
        Area under which polygon is discarded.

    filterremote : boolean


    Returns:

    ------------

    polys_simplified : geopandas polygon

    """

    if isinstance(polys, MultiPolygon):
        polys = sorted(polys, key=attrgetter('area'), reverse=True)
        mainpoly = polys[0]
        mainlength = sqrt(mainpoly.area/(2.*pi))
        if mainpoly.area > minarea:
            polys = MultiPolygon([p
                                  for p in takewhile(lambda p: p.area > minarea, polys)
                                  if not filterremote or (mainpoly.distance(p) < mainlength)])
        else:
            polys = mainpoly


    return polys





def filter_offshore_polys(offshore_polys, onshore_polys_union, minarea=0.1, filterremote=True):
    """Filter offshore polygons based on distance to main land.

    Parameters:

    ------------

    offshore_polys : geopandas polygon
        Polygon or MultiPolygon associated to a given country EEZ.

    onshore_polys_union : geopandas polygon
        Polygon or MultiPolygon of associated land-based area.

    minarea : float
        Area under which polygon is discarded.

    filterremote : boolean


    Returns:

    ------------

    polys_simplified : geopandas polygon

    """

    if isinstance(offshore_polys, MultiPolygon):
        offshore_polys = sorted(offshore_polys, key=attrgetter('area'), reverse=True)
    else:
        offshore_polys = [offshore_polys]
    mainpoly = offshore_polys[0]
    mainlength = sqrt(mainpoly.area/(5.*pi))
    polys = []
    if mainpoly.area > minarea:
        for offshore_poly in offshore_polys:

            if offshore_poly.area < minarea:
                break

            if isinstance(onshore_polys_union, Polygon):
                onshore_polys_union = [onshore_polys_union]
            for onshore_poly in onshore_polys_union:
                if not filterremote or offshore_poly.distance(onshore_poly) < mainlength:
                    polys.append(offshore_poly)
                    break
        polys = MultiPolygon(polys)
    else:
        polys = mainpoly

    return polys






def get_onshore_shapes(names, minarea=0.1, filterremote=True):
    """Load the shapes of the onshore territories a specified set of countries into a GeoPandas Dataframe.

    Parameters:

    ------------

    names : list
        List of strings, names of the countries/states of interest.

    minarea : float
        Area under which polygon is discarded.

    filterremote : boolean


    Returns:

    ------------

    onshore_shapes : geopandas dataframe
        Indexed by the name of the country and containing the shape of each country.


    """

    onshore_shapes_file_name = '../input_data/geographics/onshore_shapes.geojson'
    onshore_shapes = read_file(onshore_shapes_file_name).set_index("name")

    onshore_shapes = onshore_shapes.loc[names]
    onshore_shapes['geometry'] = onshore_shapes['geometry'].map(lambda x: filter_onshore_polys(x, minarea, filterremote))

    return onshore_shapes







def get_offshore_shapes(names, country_shapes, minarea=0.1, filterremote=False):
    """Load the shapes of the offshore territories of a specified set of countries into a GeoPandas Dataframe.

    Parameters:

    ------------

    names : list
        List of strings, names of the countries/states of interest.

    country_shapes: geopandas DataFrame
        Indexed by the name of the country and containing the shape of onshore territories of each country.

    minarea : float
        Area under which polygon is discarded.

    filterremote : boolean


    Returns:

    ------------

    offshore_shapes : geopandas dataframe
        Indexed by the name of the country and containing the shape of each country EEZ.


    """

    all_offshore_shapes_file_name = '../input_data/geographics/offshore_shapes.geojson'
    offshore_shapes = read_file(all_offshore_shapes_file_name).set_index("name")

    # Keep only associated countries
    countries_names = [name.split('-')[0] for name in names] # Allows to consider states and provinces

    offshore_shapes = offshore_shapes.reindex(countries_names)
    offshore_shapes[offshore_shapes.isna()] = Polygon([])

    country_shapes = country_shapes.loc[names]
    country_shapes_union = cascaded_union(country_shapes['geometry'].values)

    # Keep only offshore 'close' to onshore
    offshore_shapes['geometry'] = offshore_shapes['geometry'].map(lambda x: filter_offshore_polys(x,
                                                                                                  country_shapes_union,
                                                                                                  minarea,
                                                                                                  filterremote))

    return offshore_shapes






def append_database(initial_ds, appended_ds):

    """Concatenates two xarray Datasets on a given dimension.

    Parameters:

    ------------

    initial_ds : xarray.Dataset

    appeneded_ds : xarray.Dataset



    Returns:

    ------------

    concat_ds : xarray.Dataset

    """

    concat_ds = xr.concat([initial_ds, appended_ds], dim='locations')

    return concat_ds






def read_database(file_path):

    """Database reading from list of .nc files.

    Parameters:

    ------------

    file_path : str
        The path towards the .nc resource files.


    Returns:

    ------------

    dataset : xarray.Dataset
        Dataset comprising all available locations, with data covering
        a common time horizon.


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
                               chunks={'latitude': 20, 'longitude': 20})\
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







def get_global_coordinates(dataset, spatial_resolution,
                           population_density_threshold, protected_areas_selection, threshold_distance,
                           altitude_threshold, slope_threshold, forestry_threshold, depth_threshold,
                           path_population_density_data, path_protected_areas_data,
                           path_orography_data, path_land_data,
                           population_density_layer=True, protected_areas_layer=True,
                           orography_layer=True, forestry_layer=True, bathymetry_layer=True):

    """Returning the set of all available coordinate pairs.

    Parameters:

    ------------

    dataset : xarray.Dataset

    spatial_resolution : float
        Spatial resolution of the underlying grid.

    population_density_threshold : float
        Population density limit (in persons/sqkm) above which no deployments are
        possible for one given node.

    protected_areas_selection : list
        List containing the types of protected areas to be considered for removal.
        Details here:
        https://www.iucn.org/theme/protected-areas/about/protected-area-categories

    threshold_distance : float
        Distance to protected area shape under which the node is not considered
        anymore for deployments.

    altitude_threshold: float
        Altitude above which points are discarded.

    slope_threshold: float
        Terrain slope threshold above which points are discarded.

    path_population_density_data : str

    path_protected_areas_data : str

    path_orography_data : str

    path_land_data : str

    population_density_layer : boolean
        By default TRUE. It adds the population density layer, resulting in removing
        the nodes corresponding with high densities of population.

    protected_areas_layer : boolean
        By default TRUE. It adds the protected areas layer, resulting in removing the
        nodes close to those locations.

    orography_layer : boolean
        By default TRUE. It adds the orography layer, resulting in removing the
        nodes not appropriate for RES deployment based on altitude and terrain slope.

    forestry_layer : boolean
        By default TRUE. It adds the forestry layer, assuming that only a cell which is fully
        covered by forest can be discarded.

    offshore_layer : boolean
        By default TRUE. It adds the offshore layer, resulting in removing points with
        an associated depth below a given threshold.


    Returns:

    ------------

    updated_coordinates : list
        A list of tuples, all coordinate pairs available in the dataset, less the
        ones populated/protected.

    """

    start_coordinates = list(zip(dataset.longitude.values, dataset.latitude.values))

    coords_to_remove_population = []
    coords_to_remove_areas = []

    if population_density_layer:

        if spatial_resolution not in [0.5, 1.0]:
            raise ValueError(' No such resolution for the population density layer. '
                          'Run without this layer filter or update data resolution.')

        file_name = 'gpw_v4_e_atotpopbt_dens_' + str(spatial_resolution) + '.nc'
        dataset = xr.open_dataset(join(path_population_density_data, file_name))

        # Align the population grid with the reference one (i.e., from ERA5),
        # as they are shifted with 0.5 deg.
        dataset = dataset.sortby([dataset.longitude, dataset.latitude])
        dataset = dataset.assign_coords(
            longitude=arange(floor(dataset.longitude.values.min()),
                             floor(dataset.longitude.values.max()) + 1,
                             spatial_resolution),
            latitude=arange(floor(dataset.latitude.values.min()),
                            floor(dataset.latitude.values.max()) + 1,
                            spatial_resolution))
        # Rename a strange variable name
        varname = [item for item in dataset.data_vars][0]
        dataset = dataset.rename({varname: 'data'})
        # The value of 5 for "raster" fetches data for the latest estimate
        # available in the dataset, that is, 2020.
        data_pop = dataset.sel(raster=5).stack(locations=('longitude', 'latitude'))

        mask = data_pop.data.where(data_pop.data > population_density_threshold)
        coords_mask = mask[mask.notnull()].locations.values.tolist()

        coords_to_remove_population = list(set(start_coordinates).intersection(set(coords_mask)))

    if protected_areas_layer:

        R = 6371.

        file_name = 'WDPA_Feb2019-shapefile-points.shp'
        dataset = read_file(join(path_protected_areas_data, file_name))

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
                    coords_to_remove_areas.append(i)

    if orography_layer:

        file_name = 'ERA5_orography_characteristics_20181231_'+str(spatial_resolution)+'.nc'
        dataset = xr.open_dataset(join(path_orography_data, file_name))

        dataset = dataset.sortby([dataset.longitude, dataset.latitude])
        dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                        + 180) % 360) - 180)).sortby('longitude')
        dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))

        array_altitude = dataset.z / 9.80665
        array_slope = dataset.slor

        mask_altitude = array_altitude.where(array_altitude.data > altitude_threshold)
        mask_slope = array_slope.where(array_slope.data > slope_threshold)

        coords_mask_altitude = mask_altitude[mask_altitude.notnull()].locations.values.tolist()
        coords_mask_slope = mask_slope[mask_slope.notnull()].locations.values.tolist()

        coords_mask_orography = list(set(coords_mask_altitude).union(set(coords_mask_slope)))
        coords_to_remove_orography = list(set(start_coordinates).intersection(set(coords_mask_orography)))

    if forestry_layer:

        file_name = 'ERA5_surface_characteristics_20181231_'+str(spatial_resolution)+'.nc'
        dataset_land = xr.open_dataset(join(path_land_data, file_name))
        # Longitude updated from (0-360) to (-180, 180) to match resource data.
        data_land = dataset_land.assign_coords(longitude=(((dataset_land.longitude
                                                            + 180) % 360) - 180)).sortby('longitude')
        data_land = data_land.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
        array_forestry = data_land['cvh']

        mask_forestry = array_forestry.where(array_forestry.data >= forestry_threshold)
        coords_mask_forestry = mask_forestry[mask_forestry.notnull()].locations.values.tolist()

        coords_to_remove_forestry = list(set(start_coordinates).intersection(set(coords_mask_forestry)))

    if bathymetry_layer:

        file_name = 'ERA5_surface_characteristics_20181231_' + str(spatial_resolution) + '.nc'
        dataset_land = xr.open_dataset(join(path_land_data, file_name))

        data_bath = dataset_land['wmb'].assign_coords(longitude=(((dataset_land.longitude
                                                                   + 180) % 360) - 180)).sortby('longitude')

        data_bath = data_bath.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
        array_offshore = data_bath.fillna(0.)

        mask_offshore = array_offshore.where(array_offshore.data > depth_threshold)
        coords_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

        coords_to_remove_offshore = list(set(start_coordinates).intersection(set(coords_mask_offshore)))

    # Set difference between "global" coordinates and the sets computed in this function.
    updated_coordinates = set(start_coordinates) - \
                          set(coords_to_remove_population) - \
                          set(coords_to_remove_areas) - \
                          set(coords_to_remove_orography) -\
                          set(coords_to_remove_forestry) -\
                          set(coords_to_remove_offshore)

    return list(updated_coordinates)





def return_coordinates_from_countries(regions, global_coordinates, add_offshore=False):
    """Returns coordinate pairs associated with a given region. If the region
       is not pre-defined, the user is requested to input a series of tuples
       representing the vertices of a polygon defining the area of interest.

    Parameters:

    ------------

    regions : str/list
        Region for which coordinate pairs are extracted.

    global_coordinates : list
        List of all available coordinates.

    add_offshore : boolean

    Returns:

    ------------

    coordinates_dict : dictionary
        (Key, value) pairs of coordinates associated to each input region.

    """

    coordinates_dict = {}

    if isinstance(regions, str):
        regions = [regions]

    # Load countries/regions shapes
    onshore_shapes_all = read_file("../input_data/geographics/onshore_shapes.geojson")

    for region in regions:
        if region == 'EU':
            region_states = ['AT', 'BE', 'DE', 'DK', 'ES',
                                'FR', 'GB', 'IE', 'IT', 'LU',
                                'NL', 'NO', 'PT', 'SE', 'CH', 'CZ',
                                'AL', 'BA', 'BG', 'EE', 'LV', 'ME',
                                'FI', 'GR', 'HR', 'HU', 'LT',
                                'MK', 'PL', 'RO', 'RS', 'SI', 'SK']
        elif region == 'EU_W':
            region_states = ['AT', 'BE', 'DE', 'DK', 'ES',
                                'FR', 'GB', 'IE', 'IT', 'LU',
                                'NL', 'NO', 'PT', 'SE', 'CH']
        elif region == 'EU_E':
            region_states = ['AL', 'BA', 'BG', 'EE', 'LV', 'ME',
                                'FI', 'GR', 'HR', 'HU', 'LT', 'CZ',
                                'MK', 'PL', 'RO', 'RS', 'SI', 'SK']
        elif region == "ContEU":
            region_states = ['PL', 'SK', 'SI', 'HU', 'HR', 'RO', 'GR', 'RS', 'BG', 'BA', 'IT', 'ES', 'PT', 'FR',
                                'BE', 'NL', 'PT', 'ES', 'LU', 'DE', 'CZ', 'CH', 'AT']
        elif region == "CWE":
            region_states = ['DE', 'FR', 'BE', 'NL']
        elif region == 'WestEU':
            region_states = ['FR', 'BE', 'NL', 'PT', 'ES', 'LU', 'DE', 'IT', 'AT', 'CH']
        elif region == 'NA':
            region_states = ['DZ', 'EG', 'MA', 'LY', 'TN']
        elif region == 'ME':
            region_states = ['AE', 'BH', 'CY', 'IR', 'IQ', 'IL', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'SY', 'YE']
        elif region == 'US_S':
            region_states = ['US-TX']
        elif region == 'US_W':
            region_states = ['US-AZ', 'US-CA', 'US-CO', 'US-MT', 'US-WY', 'US-NM',
                             'US-UT', 'US-ID', 'US-WA', 'US-NV', 'US-OR']
        elif region == 'US_E':
            region_states = ['US-ND', 'US-SD', 'US-NE', 'US-KS', 'US-OK', 'US-MN',
                              'US-IA', 'US-MO', 'US-AR', 'US-LA', 'US-MS', 'US-AL', 'US-TN',
                              'US-IL', 'US-WI', 'US-MI', 'US-IN', 'US-OH', 'US-KY', 'US-GA', 'US-FL',
                              'US-PA', 'US-SC', 'US-NC', 'US-VA', 'US-WV',
                              'US-MD', 'US-DE', 'US-NJ', 'US-NY', 'US-CT', 'US-RI',
                              'US-MA', 'US-VT', 'US-ME', 'US-NH']
        elif region in onshore_shapes_all["name"].values:
            region_states = [region]
        else:
            raise ValueError("Unknown region ", region)

        onshore_shapes_selected = get_onshore_shapes(region_states, minarea=0.1)

        if add_offshore:

            offshore_shapes_selected = get_offshore_shapes(region_states, onshore_shapes_selected)
            all_shapes = hstack((onshore_shapes_selected["geometry"].values,
                                 offshore_shapes_selected["geometry"].values))
        else:
            all_shapes = onshore_shapes_selected["geometry"].values

        total_shape = cascaded_union(all_shapes)
        total_shape_prepped = prepared.prep(total_shape)

        points = array(global_coordinates,
                       dtype('float,float'))[
            [total_shape_prepped.contains(Point(p)) for p in global_coordinates]].tolist()

        if len(points) != 0:
            coordinates_dict[region] = points

    return coordinates_dict






def selected_data(dataset, region_coordinates, time_slice):
    """Returns dataset corresponding to a given region
    and within a given time range.

    Parameters:

    ------------

    dataset : xarray.Dataset
        Complete dataset read previously from .nc file list.

    region_coordinates : list
        Coordinates dictionary.

    time_slice : tuple
        Tuple containing start and end of time horizon to be sliced.


    Returns:

    ------------

   updated_dataset : xarray.Dataset
        Sliced dataset.

    """

    # This is a test which raised some problems on Linux distros, where for some
    # unknown reason the dataset is not sorted on the time dimension.
    if (datetime64(time_slice[0]) < dataset.time.values[0]) or \
            (datetime64(time_slice[1]) > dataset.time.values[-1]):
        raise ValueError(' At least one of the time indices exceeds the available data.')

    dataset_regions = []
    list_coordinates = []

    for key in region_coordinates.keys():
        list_coordinates.extend(list(list_chunks(region_coordinates[key], 1000)))

    # Basically, select data according to the time-space dimensions for all regions.
    for sublist_coordinates in list_coordinates:
        dataset_region = dataset.sel(locations=sublist_coordinates,
                                     time=slice(datetime64(time_slice[0]),
                                                datetime64(time_slice[1])))
        dataset_regions.append(dataset_region)

    updated_dataset = xr.concat(dataset_regions, dim='locations')

    return updated_dataset







def return_output(dataset, techs, path_to_transfer_function):
    """Returns computed hourly capacity factors.

    Parameters:

    ------------

    dataset : xarray.Dataset
        Array containing the resource indexed by location and time.

    techs : list
        Available technologies - the actual transfer function from raw resource
        to power output. Each element of the list is a str of shape
        "resource_converter". For example, for a Vestas converter for which
        the transfer function is available, one would use "wind_vestas".

    path_to_transfer_function : str
        Path to the folder containing various transfer functions.


    Returns:

    ------------

   power_output : xarray.DataArray
        Array containing the normalized output for a particular, single
        technology, indexed by location and time.


    """

    output_dict = dict.fromkeys(techs, None)

    for technology in techs:

        # Split resource and converter technology from the name string.
        resource = technology.split('_')[0]
        converter = technology.split('_')[1]

        if resource == 'wind':

            # Compute the resultant of the two wind components.
            wind = xu.sqrt(dataset.u100 ** 2 + dataset.v100 ** 2)

            coordinates = dataset.u100.coords
            dimensions = dataset.u100.dims

            if converter == 'aerodyn':

                wind_speed_references = da.from_array(arange(0.0, 35.0, 0.1), chunks='auto')
                capacity_factor_references = da.from_array(load(path_to_transfer_function
                                                  + 'wind_multi_aerodyn.npy'), chunks='auto')
                wind_data = da.from_array(wind.values, chunks='auto', asarray=True)
                power_output = da.map_blocks(interp, wind_data,
                                             wind_speed_references, capacity_factor_references).compute()

            else:
                raise ValueError(' The wind technology specified is not available yet.')



        elif resource == 'solar':

            # Get irradiance in W/m2 from J/m2
            irradiance = dataset.ssrd / 3600.
            # Get temperature in C from K
            temperature = dataset.t2m - 273.15

            coordinates = dataset.ssrd.coords
            dimensions = dataset.ssrd.dims

            if converter == 'tallmaxm':

                parameters = read_inputs(path_to_transfer_function
                                         + 'solar_tallmaxm.yml')

                c1 = parameters['c1']
                c2 = parameters['c2']
                c3 = parameters['c3']
                b = parameters['beta']
                g = parameters['gamma']
                eta_ref = parameters['eta_ref']
                t_ref = parameters['t_ref']
                G_ref = parameters['G_ref']

                irradiance.values = clip(irradiance.values, 1., None)

                # Based on https://iopscience.iop.org/article/10.1088/1748-9326/aad8f6/meta
                t_cell = c1 + c2 * temperature + c3 * irradiance
                eta_cell = eta_ref * (1 - b * (t_cell - t_ref)
                                      + xu.log10(irradiance) * g)

                power_output_array = eta_cell * irradiance / (eta_ref * G_ref)
                power_output_vector = power_output_array.values
                power_output = power_output_vector / power_output_vector.max(axis=0)

            else:
                raise ValueError(' The solar technology specified is not available yet.')

        else:
            raise ValueError(' The resource specified is not available yet.')

        output_array = xr.DataArray(power_output, coords=coordinates, dims=dimensions)
        output_dict[technology] = output_array

    return output_dict






def resource_quality_mapping(signal_dict, delta, measure):
    """Applies a rolling method to create all time windows within a given time
       horizon of interest AND computes the quality measure over each window.

    Parameters:

    ------------

    signal_dict : xarray.DataArray
        Multidimensional array indexed on location and time.

    delta : float
        Time window length.

    measure : str
        Resource quality measure (mean, percentile, etc.)


    Returns:

    ------------

    array : xarray.DataArray
        Multidimensional array indexed by location, windows.


    """

    if isinstance(delta, list):
        delta = delta[0]
    else:
        pass

    rolling_dict = dict.fromkeys(signal_dict.keys(), None)

    for tech in signal_dict.keys():

        if measure == 'mean':
            # The method here applies the mean over a rolling
            # window of length delta, centers the label and finally
            # drops NaN values resulted.
            time_array = signal_dict[tech].rolling(time=delta, center=True).mean().dropna('time')

        elif measure == 'median':
            # The method here returns the median over a rolling
            # window of length delta, centers the label and finally
            # drops NaN values resulted.
            time_array = signal_dict[tech].rolling(time=delta, center=True).median().dropna('time')

        elif measure in ['p10', 'p15', 'p20', 'p25', 'p30', 'p35',
                         'p40', 'p45', 'p55', 'p60']:

            quant = float(measure[-2:]) / 100.

            # Had to resort to some pandas method here, since xarray was
            # slower for some reason.
            df = DataFrame(data=signal_dict[tech].data)
            roll = df.rolling(window=delta, center=True, axis=0).quantile(quant).dropna()
            time_array = xr.DataArray(roll.values, coords=[arange(0, signal_dict[tech].time.shape[0] - delta + 1),
                                                           signal_dict[tech].locations], dims=['time', 'locations'])

        else:

            raise ValueError(' Measure {} is not available.'.format(str(measure)))

        # Renaming coordinate and updating its values to integers.
        time_array = time_array.rename({'time': 'windows'})
        window_dataarray = time_array.assign_coords(windows=arange(1, time_array.windows.values.shape[0] + 1))

        rolling_dict[tech] = window_dataarray

    return rolling_dict






def critical_window_mapping(rolling_signal_dict,
                            alpha_rule, alpha_plan, alpha_load_norm, alpha_numerical,
                            delta, regions, region_coordinates, date_slice, path_load_data):

    """Critical window assessment. Comparing rolling signal with alpha.

    Parameters:

    ------------

    rolling_signal_dict : xarray.DataArray
        Multidimensional array indexed on locations and windows.

    alpha_rule : str
        Rule of computing alpha.
        "Uniform" - flat value applied in space and time.
        "Percentile" - flat value applied location-wise.
        "Load-based" - time dependent, flat across regions.

    alpha_plan : str
        Planning approach chosen.
        "Centralized" - assessment of alpha is aggregated
        over all regions considered.
        "Partitioned" - alpha is assessed on a location
        by location basis.

    alpha_load_norm : str
        Normalization approach for alpha.
        "Min" - (x-xmin)/(xmax-xmin)
        "Max" - (x/xmax)

    alpha_numerical : float / str
        When alpha is not time dependent, this argument sets its value.
        If alpha_rule is "uniform", than alpha_numerical is a float (e.g., 0.5)
        If alpha_rule is "percentile", than alpha_numerical is a str (e.g., 'p50')

    delta : int
        Length of time window

    regions : list
        List of considered regions.

    region_coordinates : dict
        Coordinates of considered regions.

    date_slice : tuple
        Tuple containing start and end of time horizon to be sliced.

    path_load_data : string
        Path towards load data files.

    Returns:

    ------------

   critical_windows : xarray.DataArray
        Multidimensional array indexed by locations and windows, returning
        '0' if the window is NOT critical and '1' if IT IS critical.

    """
    # Checks which elements in rolling_signal_dict are smaller than the CF threshold.
    # In this setup, non-critical windows are 0, critical ones are 1.

    if isinstance(delta, list):
        delta = delta[0]
    else:
        pass

    critical_dict = dict.fromkeys(rolling_signal_dict.keys(), None)
    critical_array = []

    if alpha_rule == 'uniform':

        # Assessing windows for each technology in the dict
        for tech in critical_dict.keys():

            critical_windows = (rolling_signal_dict[tech] <= alpha_numerical)
            critical_array.append(critical_windows)



    elif alpha_rule == 'location_dependent':

        # Assessing windows for each technology in the dict
        for tech in critical_dict.keys():

            # Computing alpha for each location (quantile over full horizon)
            alpha_per_location = rolling_signal_dict[tech].quantile(alpha_numerical, dim='time')
            critical_windows = (rolling_signal_dict[tech] <= alpha_per_location)
            critical_array.append(critical_windows)



    elif alpha_rule == 'time_dependent':

        if alpha_plan == 'centralized':

            for tech in critical_dict.keys():

                l_norm = retrieve_load_data(path_load_data, date_slice, delta, regions,
                                            alpha_plan, alpha_load_norm)

                # Flip axes
                alpha_reference = l_norm[:, newaxis]

                critical_windows = (rolling_signal_dict[tech] <= alpha_reference)
                critical_array.append(critical_windows)

        elif alpha_plan == 'partitioned':

            for tech in critical_dict.keys():

                critical_windows_subsets = []

                # Extract lists of load subdivisions from load_dict.
                for subregion in regions:

                    l_norm = retrieve_load_data(path_load_data, date_slice, delta, subregion,
                                                alpha_plan, alpha_load_norm)

                    # Select region of interest within the dict value with 'tech' key.
                    local_filtered_signal = rolling_signal_dict[tech].sel(locations=region_coordinates[subregion])
                    # Flip axes.
                    alpha_reference = l_norm[:, newaxis]

                    proxy_array = (local_filtered_signal <= alpha_reference)
                    critical_windows_subsets.append(proxy_array)

                # Concatenate subregions back.
                critical_windows = xr.concat(critical_windows_subsets, dim='locations')
                critical_array.append(critical_windows)


    else:
        sys.exit('No such alpha rule. Retry.')

    critical_windows_overall = xr.concat(critical_array, dim='locations')

    return critical_windows_overall.astype(int)







def retrieve_load_data(path_load_data, date_slice, delta, regions, alpha_plan, alpha_load_norm):
    """Function retrieving relevant load data (time dependent alpha)
    for time-dependent critical window mapping. Load data acquired from i) ENTSO-E, on a country basis
    and ii) EIA (US Electric System Operating Data tool) on a regional basis.

    Parameters:

    ------------

    path_load data : string
        Path towards the land-sea mask files.

    date_slice : tuple
        Tuple containing start and end of time horizon to be sliced.

    delta : int
        Time window length

    regions : list
        Regions to be considered.

    alpha_plan : str

    alpha_load_norm : str



    Returns:

    ------------

    vector_load_norm : ndarray
        Vector of normalized load data.

    """

    dict_regions = {'EU': ['AT', 'BE', 'CH', 'DE', 'DK', 'ES',
                             'FR', 'GB', 'IE', 'IT', 'LU',
                             'NL', 'NO', 'PT','SE', 'CZ',
                             'BA', 'BG', 'CH', 'EE',
                             'FI', 'GR', 'HR', 'HU', 'LT', 'LV', 'ME',
                             'MK', 'PL', 'RO', 'RS', 'SI', 'SK'],
                    'EU_W': ['AT', 'BE', 'CH', 'DE', 'DK', 'ES',
                                'FR', 'GB', 'IE', 'IT', 'LU',
                                'NL', 'NO', 'PT','SE'],
                    'EU_E': ['BA', 'BG', 'CH', 'EE',
                                'FI', 'GR', 'HR', 'HU', 'LT', 'LV', 'ME',
                                'MK', 'PL', 'RO', 'RS', 'SI', 'SK'],
                    'ContEU': ['PL', 'SK', 'SI', 'HU', 'HR', 'RO', 'GR',
                               'RS', 'BG', 'BA', 'IT', 'ES', 'PT', 'FR',
                               'BE', 'NL', 'PT', 'ES', 'LU', 'DE', 'CZ',
                               'CH', 'AT'],
                    'EastEU': ['PL', 'SK', 'SI', 'HU', 'HR', 'RO', 'GR',
                               'RS', 'BG', 'BA'],
                    'SouthEU': ['IT', 'ES', 'PT'],
                    'NorthEU': ['DK', 'SE', 'NO', 'FI'],
                    'WestEU': ['FR', 'BE', 'NL', 'PT', 'ES', 'LU'],
                    'CentralEU': ['DE', 'CZ', 'CH', 'AT'],
                    'US_E': ['NEISO', 'NYISO', 'MIDATLANTIC', 'CAROLINAS',
                               'MIDWEST', 'CENTRAL', 'SOUTHEAST', 'FLORIDA', 'TENNESSEE'],
                    'TX': ['ERCOT'],
                    'US_W': ['CALIFORNIA', 'SOUTHWEST', 'NORTHWEST'],
                    'US': ['NEISO', 'NYISO', 'MIDATLANTIC', 'CAROLINAS',
                           'SOUTHEAST', 'FLORIDA', 'TENNESSEE',
                           'CALIFORNIA', 'SOUTHWEST', 'NORTHWEST',
                           'MIDWEST', 'CENTRAL', 'ERCOT'],
                    'NPCC': ['NEISO', 'NYISO'],
                    'RF': ['MIDATLANTIC'],
                    'SERC': ['CAROLINAS', 'SOUTHEAST', 'FLORIDA', 'TENNESSEE'],
                    'MRO': ['CENTRAL', 'MIDWEST'],
                    'US-ND': ['CENTRAL'],'US-SD': ['CENTRAL'],'US-NE': ['CENTRAL'],'US-KS': ['CENTRAL'],
                    'US-OK': ['CENTRAL'],'US-MN': ['MIDWEST'],'US-IA': ['MIDWEST'],'US-MO': ['MIDWEST'],
                    'US-AR': ['MIDWEST'],'US-LA': ['SOUTHEAST'],'US-MS': ['SOUTHEAST'],'US-AL': ['SOUTHEAST'],
                    'US-TE': ['TENNESSEE'],'US-IL': ['MIDWEST'],'US-WI': ['MIDWEST'],'US-MI': ['MIDWEST'],
                    'US-IN': ['MIDWEST'],'US-OH': ['MIDATLANTIC'],'US-KY': ['MIDATLANTIC'],'US-GA': ['SOUTHEAST'],
                    'US-FL': ['FLORIDA'],'US-PA': ['MIDATLANTIC'],'US-SC': ['CAROLINAS'],'US-NC': ['CAROLINAS'],
                    'US-VA': ['MIDATLANTIC'],'US-WV': ['MIDATLANTIC'],'US-MD': ['MIDATLANTIC'],'US-DE': ['MIDATLANTIC'],
                    'US-NJ': ['MIDATLANTIC'],'US-NY': ['NYISO'],'US-CT': ['NEISO'],'US-RI': ['NEISO'],'US-MA': ['NEISO'],
                    'US-VT': ['NEISO'],'US-ME': ['NEISO'],'US-NH': ['NEISO'],'US-AZ': ['SOUTHWEST'],
                    'US-CA': ['CALIFORNIA'],'US-CO': ['NORTHWEST'],'US-MT': ['NORTHWEST'],'US-WY': ['NORTHWEST'],
                    'US-NM': ['SOUTHWEST'],'US-UT': ['NORTHWEST'],'US-ID': ['NORTHWEST'],'US-WA': ['NORTHWEST'],
                    'US-NV': ['SOUTHWEST'],'US-OR': ['NORTHWEST'],'US-TX': ['ERCOT'],
                    'CWE': ['FR', 'BE', 'LU', 'NL', 'DE'],
                    'BL': ['BE', 'LU', 'NL'],
                    'ME': ['NA']}

    load_data = read_csv(join(path_load_data, 'Hourly_demand_2008_2018.csv'), index_col=0, sep=';')
    load_data.index = date_range('2008-01-01T00:00', '2017-12-31T23:00', freq='H')

    load_data_sliced = load_data.loc[date_slice[0]:date_slice[1]]

    # Adding the stand-alone regions to load dict.
    standalone_regions = list(load_data.columns)
    for region in standalone_regions:
        dict_regions.update({str(region): str(region)})

    if alpha_plan == 'centralized':

        # Extract lists of load subdivisions from load_dict.
        # e.g.: for regions ['BL', 'DE'] => ['BE', 'NL', 'LU', 'DE']
        regions_list = []
        for key in regions:
            if isinstance(dict_regions[key], str):
                regions_list.append(str(dict_regions[key]))
            elif isinstance(dict_regions[key], list):
                regions_list.extend(dict_regions[key])
            else:
                raise TypeError('Check again the type. Should be str or list.')

        load_vector = load_data_sliced[regions_list].sum(axis=1)
        load_vector_norm = return_filtered_and_normed(load_vector, delta, alpha_load_norm)

    elif alpha_plan == 'partitioned':

        load_vector = load_data_sliced[dict_regions[regions]].sum(axis=1)
        load_vector_norm = return_filtered_and_normed(load_vector, delta, alpha_load_norm)

    return load_vector_norm







def return_filtered_and_normed(signal, delta, load_norm):
    """Filters and normalizes load data (for time-dependent alpha definition).

    Parameters:

    ------------

    signal : TimeSeries

    delta : int
        Length of time window.

    load_norm : str
        Normalization approach for alpha.
        "min" - (x-xmin)/(xmax-xmin)
        "max" - (x/xmax)



    Returns:

    ------------

    l_norm : array
        Rolling and normalized profile of load.


    """

    l_smooth = signal.rolling(window=delta, center=True).mean().dropna()

    if load_norm == 'max':

        l_norm = l_smooth / l_smooth.max()

    elif load_norm == 'min':

        l_norm = (l_smooth - l_smooth.min()) / (l_smooth.max() - l_smooth.min())

    else:
        sys.exit('No such norm available. Retry.')

    return l_norm.values






def spatial_criticality_mapping(data_array):
    """Sums up critical locations per each time window.

    Parameters:

    ------------

    data_array : xarray.DataArray
        Multidimensional array indexed on locations and windows.



    Returns:

    ------------

   spatial_criticality : xarray.DataArray
        Array indexed on windows.

    """
    # Computes the spatial criticality of each windows. For one window, it sums
    # up values on the 'location' dimension and divides the result by the length
    # of the same dimension.
    spatial_criticality = data_array.sum(dim='locations') / data_array.locations.values.shape[0]

    return spatial_criticality






def spatiotemporal_criticality_mapping(data_array, beta):
    """Criticality indicator computation.

    Parameters:

    ------------

    data_array : xarray.DataArray
        Array indexed on windows.

    beta : float
        Geographical coverage threshold.




    Returns:

    ------------

   spatiotemporal_criticality : float
        Value of the criticality indicator of the region.

    """
    # Checks beta-criticality (returns 1 if critical, 0 otherwise) and computes
    # the criticality index for a given region by dividing the sum on dimension
    # 'windows' to its length.
    spatiotemporal_criticality_proxy = (data_array >= beta)
    spatiotemporal_criticality = spatiotemporal_criticality_proxy.sum(dim='windows') \
                                 / data_array.windows.values.shape[0]

    return spatiotemporal_criticality.values






def get_matrix_indices(technologies, deployments, coordinates_dict):
    """Returns indices of locations within the optimization matrix (treat this carefully!).

    Parameters:

    ------------

    technologies : list
        List of technologies.

    deployments : list
        List of deployments per region.

    coordinates_dict : dict
        Dictionary of coordinate pairs associated with each input region.



    Returns:

    ------------

    k : float
        Number of regions.

    indices_as_list : list
        Boundary indices (start-end) of locations (taking also into account
        different conversion technologies within the optimization matrix.

    """

    if isinstance(deployments, list):

        indices_as_list = []
        indices = {}
        k = len(deployments)

        if len(deployments) == 1:
            # Setting left bound to 1 and
            # right bound to (total_no_locations*no_technologies + 1)
            bounds = [1, sum(len(coordinates_dict[key])
                             for key in coordinates_dict.keys()) * len(technologies) + 1]
            indices_as_list.append(list(arange(bounds[0], bounds[1], 1)))

        elif len(deployments) == len(coordinates_dict):
            for tech in technologies:

                for region in list(coordinates_dict.keys()):

                    incumbent_key = str(tech) + '_' + str(region)

                    # If the dict is empty...
                    if not indices:
                        # Indices of the first (tech, region) group.
                        indices[incumbent_key] = list(arange(1, len(coordinates_dict[region]) + 1))
                        last_key = incumbent_key

                    else:
                        # Add indices in continuation.
                        start_index = indices[last_key][-1] + 1
                        indices[incumbent_key] = list(
                            arange(start_index, start_index + len(coordinates_dict[region]), 1))
                        last_key = incumbent_key

            # Convert the dict into a list of lists.
            for region in list(coordinates_dict.keys()):
                indices_per_region = []
                for item in indices:
                    if region in item:
                        indices_per_region.extend(indices[item])
                indices_as_list.append(indices_per_region)

        else:
            raise ValueError(' Number of regions does not match number of partitions.')

    else:
        raise ValueError(' Wrong variable type for the partitions.')

    return k, indices_as_list






def build_parametric_dataset(path_landseamask, path_population, path_buses,
                             coordinates_dict, technologies, spatial_resolution, distance_assessment=False):

    """Function assessing capacity potential, cost estimates, distance from buses.

    Parameters:

    ------------

    path_landseamask : string
        Path towards the land-sea mask files.

    path_population : string
        Path towards the population files.

    path_buses : string
        Path towards the files containing existing network data.

    coordinates_dict : dict
        Dictionary of coordinate pairs associated with each input region.

    technologies : list
        List containing all considered technologies.

    spatial_resolution : float
        Spatial resolution of the data.


    Returns:

    ------------

    capacity_array : array
        Array containing installed capacity potential for all
        locations and technologies.

    cost_array : array
        Array containing cost estimates for all locations and
        technologies. Currently, the cost estimation does not refer
        to an economic value, but to some relative weights across
        different locations (e.g., if wind onshore costs 1, the
        offshore cost would be 2).

    distance_from_grid : array
        Array containing the Euclidian distance from the set of
        potential sites to existing electrical infrastructure.

    """

    data = xr.Dataset()

    dataset_land = xr.open_dataset(join(path_landseamask,
                                        'ERA5_surface_characteristics_20181231_'
                                        + str(spatial_resolution) + '.nc'))
    # Longitude updated from (0-360) to (-180, 180) to match resource data.
    data_land = dataset_land.assign_coords(longitude=(((dataset_land.longitude
                                                        + 180) % 360) - 180)).sortby('longitude')
    data_land = data_land.drop('time').squeeze()
    # Land-sea mask data.
    data['lsm'] = data_land['lsm']
    # Vegetation mask data.
    data['cvl'] = data_land['cvl']
    # Account for possibility of deployments in high-vegetation areas too.
    data['cvh'] = data_land['cvh'].clip(None, 0.75)
    # Normalize bathymetry data
    data['wmb'] = (data_land['wmb'] / data_land['wmb'].max()).fillna(0.)

    dataset_pop = xr.open_dataset(join(path_population, 'gpw_v4_e_atotpopbt_dens_'
                                       + str(spatial_resolution) + '.nc'))
    dataset_pop = dataset_pop.sortby([dataset_pop.longitude, dataset_pop.latitude])
    # Update coordinates of the population dataset to match the resource dataset.
    # Basically, the (lon, lat) of the original dataset were sampled at .5 values,
    # Now, they are shifted to .0 values.
    dataset_pop = dataset_pop.assign_coords(
        longitude=arange(floor(dataset_pop.longitude.values.min()),
                         floor(dataset_pop.longitude.values.max()) + 1,
                         spatial_resolution),
        latitude=arange(floor(dataset_pop.latitude.values.min()),
                        floor(dataset_pop.latitude.values.max()) + 1,
                        spatial_resolution))

    # Rename a strange variable name
    varname = [item for item in dataset_pop.data_vars][0]
    dataset = dataset_pop.rename({varname: 'data'})
    # The value of 5 for "raster" fetches data for the latest estimate available in the dataset: 2020.
    data_pop = dataset['data'].sel(raster=5)
    data_pop = data_pop.drop('raster').squeeze()
    data['dpop'] = data_pop
    data['dpop'] = data['dpop'].fillna(0.).clip(1.).round(1)

    R = 6371.
    longitudes = data_land.longitude
    latitudes = data_land.latitude

    lon_1 = radians(longitudes - spatial_resolution / 2)
    lon_2 = radians(longitudes + spatial_resolution / 2)
    lat_1 = radians(latitudes - spatial_resolution / 2)
    lat_2 = radians(latitudes + spatial_resolution / 2)

    # Compute area for each grid cell.
    d = (sin(lat_2) - sin(lat_1)) * (lon_2 - lon_1) * R ** 2
    data['area'] = d.round(1)

    if distance_assessment:

        full_path_bus_data = join(path_buses, 'buses_all.csv')
        # Read network topology data.
        bus = read_csv(full_path_bus_data, sep=';', index_col=0)

        # Define set of bus coordinates.
        bus_coordinate_array = array(list(zip(bus.x, bus.y)))
        # Define a 2D array with all grid points.
        locations_coordinate_array = array(list(data.stack(locations=('longitude', 'latitude')).locations.values))

        # Compute distance to the buses for the full regular grid. Adds up to cost estimation.
        distance_from_grid_full = distance.cdist(locations_coordinate_array, bus_coordinate_array).min(axis=1)

        df_distance = DataFrame(index=MultiIndex.from_tuples(
                                                data.stack(locations=('longitude', 'latitude')).locations.values,
                                                names=('longitude', 'latitude')
                                                ),
                                          data=distance_from_grid_full, columns=['dist_grid'])


        # Add distance to grid to the dataset.
        data = data.merge(df_distance.to_xarray(), join='left')

        # Normalize data
        data['dist_grid'] = data['dist_grid'] / data['dist_grid'].max()

    else:

        data['dist_grid'] = 0.

    # Base capacity share (1% of area) for wind deployments. Source: Brown et al. (2018).
    wind_cap_mult_by_area = 0.01
    # Some artificial additional scaling factor for wind.
    descale_wind = 1.0
    # Base capacity share (10% of area) for solar deployments. Source: Brown et al. (2018)
    pv_cap_mult_by_area = 0.1
    # Some artificial additional scaling factor for solar PV.
    descale_pv = 0.025

    # Functions estimating the capacity potential of wind and solar for each node.
    data['capacity_wind'] = (data['area'] * (1 - data['cvh']) * (
                1 / (data['dpop'] ** 0.8)) * wind_cap_mult_by_area * descale_wind).round(0)
    data['capacity_pv'] = (data['area'] * (1 - data['cvh']) * (
                1 / (data['dpop'] ** 0.25)) * pv_cap_mult_by_area * descale_pv).round(0)

    base_cost_wind = 2.
    base_cost_pv = 1.

    # Functions estimating the cost of wind and solar development for each node.
    data['cost_wind'] = (base_cost_wind * (2 - data['lsm']) * (1 + data['wmb']) * (1 + data['dist_grid']))
    data['cost_pv'] = (base_cost_pv * xu.square(2 - data['lsm']) * (1 + data['wmb']) * (1 + data['dist_grid'])) ** 1.2

    list_coords = []
    for item in coordinates_dict.items():
        list_coords.extend(item[1])

    dataset = data.stack(locations=('longitude', 'latitude')).sel(locations=list_coords)

    capacity_array = []
    cost_array = []

    # Build associated vector for the optimization framework.
    for item in technologies:
        if 'wind' in item:
            capacity_array.extend(dataset.capacity_wind.values)
            cost_array.extend(dataset.cost_wind.values)
        elif 'solar' in item:
            capacity_array.extend(dataset.capacity_pv.values)
            cost_array.extend(dataset.cost_pv.values)



    return asarray(capacity_array), asarray(cost_array)






def retrieve_optimal_locations(model_instance, window_data, technologies, problem):
    """Retrieval of coordinate pairs associated with the optimal location.

    Parameters:

    ------------

    model_insance : pyomo instance

    window_data : xarray.DataArray
        Critical window matrix.

    technologies : list
        List of considered technologies.

    problem : str
        Type of problem that is solved.



    Returns:

    ------------

    location_list : list of tuples


    """

    location_dict = dict.fromkeys(technologies, None)
    no_locations = len(model_instance.L) / len(technologies)

    for i, key in enumerate(location_dict.keys()):

        location_list = []

        start_index = i * no_locations + 1
        end_index = (i + 1) * no_locations

        if problem == 'Covering':

            for loc in arange(start_index, end_index + 1, dtype=int):
                if model_instance.x[loc].value == 1.0:
                    location_list.append(window_data.isel(locations=loc - 1).locations.values.flatten()[0])
            location_dict[key] = location_list

        elif problem == 'Load':

            for loc in arange(start_index, end_index + 1, dtype=int):
                if model_instance.x[loc].value != 0.0:
                    location_list.append(window_data.isel(locations=loc - 1).locations.values.flatten()[0])
            location_dict[key] = location_list

        else:

            raise ValueError(' This problem does not exist.')

    return location_dict








def retrieve_deployed_capacities(model, technologies, capacity_array):
    """Retrieval of deployed capacities associated with
    the optimal location (in the capacity-based case).

    Parameters:

    ------------

    model_insance : pyomo instance

    technologies : list
        List of considered technologies.

    capacity_array : array
        Array of capacity potentials per location.



    Returns:

    ------------

    capacities : dict
        Dict with capacities associated with every (tech, node) pair.


    """
    no_locations = int(len(model.L) / len(technologies))
    capacities = dict.fromkeys(technologies, None)

    idx = 1
    for key in capacities.keys():
        capacities[key] = array([model.x[i].value for i in range(idx, idx + no_locations)]) * capacity_array[idx - 1:idx + no_locations - 1]
        idx += no_locations

    return capacities






def dist_to_grid_as_penalty(path_buses, coordinates_dict, technologies):
    """Computes distance from subset of points within the regular grid representing
    the regions of interest to the existing buses.

    Parameters:

    ------------

    path_buses : str
        Path to buses file.

    coordinates_dict : dict
        Dictionary of coordinates for regions of interest.

    technologies : list
        List of technologies of interest.



    Returns:

    ------------

    distance_from_grid_norm : array
        Normalized distance (Euclidean) from candidate locations to buses.

    """

    full_path_bus_data = join(path_buses, 'buses_all.csv')
    # Read network topology data.
    bus = read_csv(full_path_bus_data, sep=';', index_col=0)

    coordinate_list = []
    for item in coordinates_dict.values():
        coordinate_list.extend(item)
    # Define set of potential locations.
    site_coordinate_array = array(coordinate_list)
    # Define set of bus coordinates.
    bus_coordinate_array = array(list(zip(bus.x, bus.y)))

    # Compute Euclidean distance between the two sets.
    distance_from_grid_per_tech = distance.cdist(site_coordinate_array, bus_coordinate_array).min(axis=1)
    # Tile the array to account for multiple technologies. Also, compute the inverse so it fits
    # into the maximization framework (smaller distance accounts for larger "penalty" term).
    distance_from_grid = 1 / tile(distance_from_grid_per_tech, len(technologies))
    distance_from_grid_norm = distance_from_grid / max(distance_from_grid)

    return distance_from_grid_norm









def apply_rolling_transformations(output_data, load_array, delta):
    """Computes rolling transformation on the capacity factor matrix for the load-based formulation.

        Parameters:

        ------------

        output_data : ndarray
            Capacity factor matrix.

        load_array : ndarray
            Load data matrix.

        delta : int
            Length of time window.



        Returns:

        ------------

        outputs : dict
            Dictionary with transformed matrices.

        """

    u_list = []
    for item in output_data.keys():
        u_list.append(output_data[item].values)
    u = concatenate(tuple(u_list), axis=1)

    u_list = []
    load_list = []

    u_dataframe = DataFrame(data=u)
    load_dataframe = DataFrame(data=load_array)

    for d in delta:
        u_rolling = u_dataframe.rolling(window=d, center=True, axis=0).mean().dropna()
        u_list.append(u_rolling)
        load_rolling = load_dataframe.rolling(window=d, center=True, axis=0).mean().dropna()
        load_list.append(load_rolling)

    u_hat = concatenate(tuple(u_list), axis=0)
    load_array_hat = concatenate(tuple(load_list), axis=0)

    outputs = {'aggregated_capacity_factors': u_hat,
               'aggregated_load_array': load_array_hat}

    return outputs











def retrieve_y_idx(instance, share_random_keep=0.2):
    """Selecting constraints to be dualised.

        Parameters:

        ------------

        instance : pyomo ConcreteModel

        share_random_keep : float
            Between 0 and 1, defines the number of constraints to be randomly selected.


        Returns:

        ------------

        y_idx_dual, y_idx_keep : list
            Lists of constraints to i) be dualised and ii) be kept for BB.

        """

    all_ys = array(list(instance.y.extract_values().values()))

    y_idx_dual = []

    for i, item in enumerate(all_ys):
        if i == 0:
            if item == 1.0:
                y_idx_dual.append(i + 1)
        elif i == (len(all_ys) - 1):
            if item == 1.0:
                y_idx_dual.append(i + 1)
        else:
            if item == 1.0:
                if not ((all_ys[i - 1] != 1.0) or (all_ys[i + 1] != 1.0)):
                    y_idx_dual.append(i + 1)

    y_idx_keep = list(set(instance.W) - set(y_idx_dual))

    share = share_random_keep
    add_constraints = sample(y_idx_dual, int(share*len(y_idx_dual)))
    y_idx_keep.extend(add_constraints)
    y_idx_dual = list(set(y_idx_dual).difference(set(add_constraints)))

    custom_log(' Number of constraints kept: {}'.format(len(y_idx_keep)))

    return y_idx_dual, y_idx_keep









def concatenate_and_filter_arrays(list_of_arrays, ys_keep):
    """Concatenate a list of 1d arrays and keep only rows associated with ys that are dualized.

        Parameters:

        ------------

        list_of_arrays : list
            List of 1d arrays.

        ys_keep : list
            List of contraint indices kept in the problem.


        Returns:

        ------------

        ndarray: array

        """

    ndarray = delete(vstack(list_of_arrays), [i - 1 for i in ys_keep], axis=1)

    return ndarray










def build_init_multiplier(ys_dual, range=0.5):
    """Function initializing the Langrangian multiplier.

        Parameters:

        ------------

        ys_dual : list
            List constraints to be dualised.

        range : float
            Float determining the range in which the multiplier is defined


        Returns:

        ------------

        dict_values: dict
            Dictionary containing (index, value) pairs of the Lagrange multiplier.

        """

    values = clip(uniform(-range, range, size=len(ys_dual)), 0., range)

    dict_values = dict(zip(ys_dual, values))

    return dict_values










def retrieve_next_multiplier(instance, init_multplier, y_idx_keep, y_idx_dual,
                             iter_no, iter_total, subgradient_method,
                             a = 0.1, share_of_iter=1.0):
    """Update of the Lagrange multiplier.

        Parameters:

        ------------

        instance : pyomo ConcreteModel

        init_multiplier : array
            Incumbent Lagrange multipler.

        y_idx_keep: list
            List of indices of the kept constraints.

        y_idx_dual: list
            List of indices of dualised constraints.

        iter_no: int
            Iteration counter.

        iter_total: int
            Number of total iterations.

        subgradient_method: str
            Subgradient method applied (Inexact/Exact). Influences the stepping policy.

        a: float
            Paramter of the stepping policy

        share_of_iter: float
            Between 0.0 and 1.0, used in the "Inexact" method and defines the beginning of non-constant stepping.


        Returns:

        ------------

        multiplier: array
            Updated Lagrange multiplier.

        """
    d = delete(instance.D, [i - 1 for i in y_idx_keep], axis=0)

    x_array = array(list(instance.x.extract_values().values()))
    y_array = take(array(list(instance.y.extract_values().values())), [i - 1 for i in y_idx_dual])

    if subgradient_method == 'Inexact':

        count = 1
        share = share_of_iter
        iter_threshold = share * iter_total

        if iter_no < iter_threshold:
            eta = a
        else:
            eta = a / (count + 1)
            count += 1

    elif subgradient_method == 'Exact':
        eta = a*1e2 / (iter_no + 1)

    else:
        raise ValueError(' This subproblem is not available.')

    sgr = dot(d, x_array) - multiply(instance.c, y_array)

    multiplier_values = array(list(init_multplier.values())) - multiply(eta, sgr)
    multiplier_positive = where(multiplier_values < 0., 0., multiplier_values)

    multiplier = dict(zip(y_idx_dual, multiplier_positive))

    return multiplier





def new_cost_rule(instance, y_idx_keep, y_idx_dual, multiplier, low_memory=False):
    """Update of the objective function, based on the new multiplier value.
        Parameters:

        ------------

        instance : pyomo ConcreteModel

        multiplier : array
            Incumbent Lagrange multipler.

        y_idx_keep: list
            List of indices of the kept constraints.

        y_idx_dual: list
            List of indices of dualised constraints.

        low_memory: boolean
            If True, pypsa framework used for objective construction.


        Returns:

        ------------

        instance.objective: pyomo object
            Updated objective function.

        """
    if low_memory == True:

        instance.objective = Objective(expr=sum(instance.y[w] for w in instance.W) + \
                                            sum(multiplier[w] * (
                                                    sum(instance.D[w - 1, l - 1] * instance.x[l] for l in
                                                        instance.L) - instance.c * instance.y[w]) for w in y_idx_dual),
                                       sense=maximize)



    else:

        lc = dict(zip(y_idx_dual, multiply(instance.c, array(list(multiplier.values())))))
        dx = dot(array(list(multiplier.values())),
                 delete(instance.D, [i - 1 for i in y_idx_keep], axis=0))

        objective = LExpression()
        objective.variables.extend([(1, instance.y[w]) for w in instance.W])
        objective.variables.extend([dx[l - 1], instance.x[l]] for l in instance.L)
        objective.variables.extend([(-lc[w], instance.y[w]) for w in y_idx_dual])

        instance.objective = Objective(expr=0., sense=maximize)
        instance.objective._expr = _build_sum_expression(objective.variables)

    return instance.objective




def new_cost_rule_penalty(input_data, instance, y_idx_keep, y_idx_dual, multiplier, low_memory=False):

    critical_windows = input_data['critical_window_matrix']
    no_locations = critical_windows.shape[1]
    no_windows = critical_windows.shape[0]

    n = input_data['number_of_deployments']

    no_critwind_per_location = full(no_locations, float(no_windows)) - instance.D.sum(axis=0)
    lamda = 1e-3 / sum(n)
    penalty = multiply(lamda, no_critwind_per_location)


    if low_memory == True:

        instance.objective = Objective(expr=sum(instance.y[w] for w in instance.W) + \
                                            sum(instance.x[l] * penalty[l - 1] for l in instance.L) + \
                                            sum(multiplier[w] * (
                                            sum(instance.D[w - 1, l - 1] * instance.x[l] for l in instance.L) - instance.c * instance.y[w]) for w in y_idx_dual),
                                       sense=maximize)



    else:

        lc = dict(zip(y_idx_dual, multiply(instance.c, array(list(multiplier.values())))))
        dx = dot(array(list(multiplier.values())),
                 delete(instance.D, [i - 1 for i in y_idx_keep], axis=0))

        objective = LExpression()
        objective.variables.extend([(1, instance.y[w]) for w in instance.W])
        objective.variables.extend([(penalty[l - 1], instance.x[l]) for l in instance.L])
        objective.variables.extend([dx[l - 1], instance.x[l]] for l in instance.L)
        objective.variables.extend([(-lc[w], instance.y[w]) for w in y_idx_dual])

        instance.objective = Objective(expr=0., sense=maximize)
        instance.objective._expr = _build_sum_expression(objective.variables)

    return instance.objective







def generate_neighbor(x, N):
    """Generation of neighbours in the Simulated Annealing algorithm.
        Parameters:

        ------------

        x : array
            Initial solution.

        N : int
            Number of swaps (i.e., the Hamming distance).


        Returns:

        ------------

        x_array: array
            Updated solution.

        """
    x_array = array(list(x))

    ones = where(x_array == 1)[0]
    zero = where(x_array == 0)[0]

    x_array[nr.choice(ones, size=N, replace=False)] = 0.
    x_array[nr.choice(zero, size=N, replace=False)] = 1.

    return x_array





def simulated_annealing_epoch(no_windows, xs_incumbent, D, c, N, T, i):
    """The Simulated Annealing process.
        Parameters:

        ------------

        no_windows: int
            Number of time windows

        xs_incumbent: array
            Incumbent solution.

        D: array
            Criticality matrix.

        c: int
            Real parameter in the optimisation problem.

        N: int
            The Hamming distance within the SA algorithm.

        T: float
            Temperature, as defined in the SA algorithm.

        i: int
            Iteration count. Used solely in the multiprocessing set-up.


        Returns:

        ------------

        xs_inucmbent, sum(ys_incumbent): array, int
            Updated integer solution and its score within the SA algorithm.

        """
    ys_incumbent = ones(no_windows)
    ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.

    xs_next = generate_neighbor(xs_incumbent, N)

    ys_next = ones(no_windows)
    ys_next[where(sum(multiply(D, xs_next), axis=1) < multiply(c, ys_next))] = 0.

    delta = sum(ys_next) - sum(ys_incumbent)

    if delta > 0.:

        xs_incumbent = xs_next

    else:

        if nr.binomial(n=1, p=exp(delta / T)) == 1.:

            xs_incumbent = xs_next

    ys_incumbent = ones(no_windows)
    ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.

    return xs_incumbent, sum(ys_incumbent)







def retrieve_lower_bound(input_data, instance, method, multiprocess=True,
                                 N = 2, T_init=100., no_iter = 1000, no_epoch = 50):
    """Computing the LB of the problem.
        Parameters:

        ------------

        input_data: dict
            Dictionary containing various strucures with problem-related information.

        instance: pyomo ConcreteModel

        method: str
            Method deployed to compute the LB.
            Choices are: the simple "Projection" or the more intensive "SimulatedAnnealing"

        multiprocess: boolean
            Whether to deploy or not multiprocessing capabilities. Worth only on (very) large SA set-ups.

        N: int
            The Hamming distance.

        T_init: float
            Temperature, as defined in the SA algorithm.

        no_iter: int
            Number of epochs within the SA.

        no_epoch: int
            Number of computations within one epoch.


        Returns:

        ------------

        lower_bound: int
            Updated lower bound of the problem.

        """
    critical_windows = input_data['critical_window_matrix']
    no_locations = critical_windows.shape[1]
    no_windows = critical_windows.shape[0]
    beta = input_data['geographical_coverage']
    n = input_data['number_of_deployments']

    D = ones((no_windows, no_locations)) - critical_windows
    c = int(floor(sum(n) * round((1 - beta), 2)) + 1)

    xs_incumbent = array(list(instance.x.extract_values().values()))

    if method == 'Projection':

        ys_incumbent = ones(no_windows)
        ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.

    elif method == 'SimulatedAnnealing':

        if N > sum(n):
            raise ValueError(' Number of swaps greater than the cardinality.')

        ys_init = ones(no_windows)
        ys_init[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_init))] = 0.

        T = T_init
        a = 0.95

        if multiprocess == True:

            with Parallel(n_jobs=-1) as p:

                for it in tqdm(arange(1, no_iter + 1), desc='Simulated Annealing Loop'):

                    results_mp = p(delayed(simulated_annealing_epoch)(no_windows, xs_incumbent, D, c, N, T, i)
                                                                                    for i in arange(no_epoch))

                    xs_incumbent = choice([x[0] for x in results_mp if x[1] == max([x[1] for x in results_mp])])

                    T *= a

                ys_incumbent = ones(no_windows)
                ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.

        else:

            for it in tqdm(arange(1, no_iter + 1), desc='Simulated Annealing Loop'):

                for iter in arange(no_epoch):

                    ys_incumbent = ones(no_windows)
                    ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.

                    xs_next = generate_neighbor(xs_incumbent, N)

                    ys_next = ones(no_windows)
                    ys_next[where(sum(multiply(D, xs_next), axis=1) < multiply(c, ys_next))] = 0.

                    delta = sum(ys_next) - sum(ys_incumbent)

                    if delta > 0.:

                        xs_incumbent = xs_next

                    else:

                        if nr.binomial(n=1, p=exp(delta/T)) == 1.:

                            xs_incumbent = xs_next

                    if iter == no_epoch - 1:

                        ys_incumbent = ones(no_windows)
                        ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.

            T *= a

    if sum(ys_init) > sum(ys_incumbent):
        ys_incumbent = ys_init

    custom_log(' Lower bound improved by {} windows. LB = {}.'.format(sum(ys_incumbent) - sum(ys_init), sum(ys_incumbent)))

    return sum(ys_incumbent)










def retrieve_upper_bound(instance_results):
    """Retrieval of the UB from the relaxation.
    Parameters:

    ------------

    instance_results : dict
        Dictionary containing problem result.

    Returns:

    ------------

    upper_bound: float
        Upper bound value.

    """

    upper_bound = round(instance_results['Problem'][0]['Upper bound'], 2)

    return upper_bound








# def retrieve_lower_bound(input_data, instance):
#
#     critical_windows = input_data['critical_window_matrix']
#     no_locations = critical_windows.shape[1]
#     no_windows = critical_windows.shape[0]
#     beta = input_data['geographical_coverage']
#     n = input_data['number_of_deployments']
#
#     D = ones((no_windows, no_locations)) - critical_windows.values
#     c = int(floor(sum(n) * round((1 - beta), 2)) + 1)
#
#     xs = array(list(instance.x.extract_values().values()))
#     ys = ones((no_windows))
#
#     ys[where(sum(multiply(D, xs), axis=1) < multiply(c, ys))] = 0
#
#     return sum(ys)







# def retrieve_lower_bound_sa(input_data, instance, N, T_init = 100., epoch_count=20, iter_count=100):
#
#     critical_windows = input_data['critical_window_matrix']
#     no_locations = critical_windows.shape[1]
#     no_windows = critical_windows.shape[0]
#     beta = input_data['geographical_coverage']
#     n = input_data['number_of_deployments']
#
#     if N > sum(n):
#         raise ValueError(' Number of swaps greater than the cardinality.')
#
#     D = ones((no_windows, no_locations)) - critical_windows.values
#     c = int(floor(sum(n) * round((1 - beta), 2)) + 1)
#
#     xs_incumbent = array(list(instance.x.extract_values().values()))
#
#     ys_init = ones((no_windows))
#     ys_init[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_init))] = 0.
#     init_lb = sum(ys_init)
#
#     T = T_init
#     a = 0.95
#
#     for i in tqdm(arange(1, iter_count + 1), desc='Simulated Annealing Loop'):
#
#         for iter in arange(epoch_count):
#
#             ys_incumbent = ones((no_windows))
#             ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.
#
#             xs_next = generate_neighbor(xs_incumbent, N)
#
#             ys_next = ones((no_windows))
#             ys_next[where(sum(multiply(D, xs_next), axis=1) < multiply(c, ys_next))] = 0.
#
#             delta = sum(ys_next) - sum(ys_incumbent)
#
#             if delta > 0.:
#
#                 xs_incumbent = xs_next
#
#             else:
#
#                 if nr.binomial(n=1, p=exp(delta/T)) == 1.:
#
#                     xs_incumbent = xs_next
#
#
#         if i == iter_count - 1:
#
#             ys_incumbent = ones((no_windows))
#             ys_incumbent[where(sum(multiply(D, xs_incumbent), axis=1) < multiply(c, ys_incumbent))] = 0.
#
#         T *= a
#
#
#     custom_log(' Lower bound improved by {} windows.'.format(sum(ys_incumbent) - init_lb))
#
#     if init_lb > sum(ys_incumbent):
#
#         ys_incumbent = ys_init
#
#     return sum(ys_incumbent)








def retrieve_feasible_solution_projection(input_data, instance, instance_results, problem):
    """Computation of a feasible solution for the "Projection" solution method.
        Parameters:

        ------------

        input_data: dict
            Dictionary containing various strucures with problem-related information.

        instance: pyomo ConcreteModel

        instance_results: dict
            Results of instance.

        problem: str
            Selected problem class.

        Returns:

        ------------

        """

    if problem == 'Covering':

        critical_windows = input_data['critical_window_matrix']
        no_locations = critical_windows.shape[1]
        no_windows = critical_windows.shape[0]
        beta = input_data['geographical_coverage']
        n = input_data['number_of_deployments']

        D = ones((no_windows, no_locations)) - critical_windows.values
        c = int(floor(sum(n) * round((1 - beta), 2)) + 1)

        xs = array(list(instance.x.extract_values().values()))
        ys = ones((no_windows))
        ys[where(sum(multiply(D, xs), axis=1) < multiply(c, ys))] = 0.

        ub = round(instance_results['Problem'][0]['Upper bound'], 2)
        lb = sum(ys)
        gap = round((ub - lb) / lb * 100., 2)

        custom_log(' UB = {}, LB = {}, gap = {}%'.format(ub, lb, gap))



    elif problem == 'Load':

        delta = input_data['time_window']

        output_data = input_data['capacity_factors_dict']
        load_array = input_data['load_centralized']
        capacity_array = input_data['capacity_potential_per_node']

        u_hat = apply_rolling_transformations(output_data, load_array, delta)['aggregated_capacity_factors']
        load_array_hat = apply_rolling_transformations(output_data, load_array, delta)['aggregated_load_array']

        no_windows = u_hat.shape[0]
        W = arange(1, no_windows + 1)

        xs = array(list(instance.x.extract_values().values()))

        # y = ones((no_windows))
        # W = arange(1, no_windows + 1)
        # L = arange(1, no_locations + 1)
        #
        # for w in W:
        #
        #     if not sum(u_hat[w - 1, l - 1] * capacity_array[l - 1] * xs[l - 1] for l in L) \
        #                                                                         >= load_array_hat[w - 1] * y[w - 1]:
        #         y[w - 1] = 0.

        ys = ones(no_windows)
        ys[where(sum(multiply(multiply(u_hat, capacity_array), xs), axis=1) < multiply(load_array_hat, ys))] = 0.

        ub = round(instance_results['Problem'][0]['Upper bound'], 2)
        lb = sum(ys[w - 1] for w in W)
        gap = round((ub - lb) / lb * 100., 2)

        custom_log(' UB = {}, LB = {}, gap = {}%'.format(ub, lb, gap))

    else:

        raise ValueError(' No such problem.')

    return None
















# def retrieve_lagrangian_multipliers(input_data, model):
#
#     duals = array([model.dual[model.criticality_activation_constraint[w]] for w in model.W])
#     indices = arange(1, input_data['critical_window_matrix'].shape[0] + 1)
#     dual_array = list(zip(indices, duals))
#
#     selected = [x for (x, y) in dual_array if y != 0.0]
#
#     return {'all_duals': duals,
#             'selected_duals': selected}












def init_folder(keepfiles):
    """Initilize an output folder.

    Parameters:

    ------------

    keepfiles : boolean
        If False, folder previously built is deleted.

    Returns:

    ------------

    path : str
        Relative path of the folder.


    """

    date = strftime("%Y%m%d")
    time = strftime("%H%M%S")

    if not isdir("../output_data"):
        makedirs(abspath("../output_data"))

        path = abspath('../output_data/' + str(date) + '_' + str(time))
        makedirs(path)

    else:
        path = abspath('../output_data/' + str(date) + '_' + str(time))
        makedirs(path)

    custom_log(' Folder path is: {}'.format(str(path)))

    if keepfiles == False:
        custom_log(' WARNING! Files will be deleted at the end of the run.')

    return path






def remove_garbage(keepfiles, output_folder, lp=True, script=True, sol=True):

    """Remove different files after the run.

    Parameters:

    ------------

    keepfiles : boolean
        If False, folder previously built is deleted.

    output_folder : str
        Path of output folder.

    """

    if keepfiles == False:
        rmtree(output_folder)

    directory = getcwd()

    if lp == True:
        files = glob(join(directory, '*.lp'))
        for f in files:
            remove(f)

    if script == True:
        files = glob(join(directory, '*.script'))
        for f in files:
            remove(f)

    if sol == True:
        files = glob(join(directory, '*.sol'))
        for f in files:
            remove(f)






def read_inputs(inputs):

    """Reading data from .yml file

    Parameters:

    ------------

    inputs : str
        Path towards the parameter file.

    Returns:

    ------------

    data : dict
        Dict containing parameters.


    """
    with open(inputs) as infile:
        data = yaml.safe_load(infile)

    return data






def custom_log(message):

    print(datetime.now().strftime('%H:%M:%S')+' --- '+str(message))
