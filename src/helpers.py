from copy import deepcopy
from datetime import datetime
from glob import glob
from itertools import takewhile
from operator import attrgetter
from os import remove, getcwd, makedirs
from os.path import join, isdir, abspath
from shutil import rmtree

import xarray as xr
import yaml
from geopandas import read_file
from numpy import sqrt, hstack, arange, dtype, array, timedelta64
from pandas import DataFrame, read_excel, notnull, read_csv, date_range, to_datetime
from scipy.spatial import distance
from shapely import prepared
from shapely.geometry import Point, MultiPolygon, Polygon
from shapely.ops import unary_union
from xarray import concat


def filter_onshore_polys(polys, minarea=0.1, filterremote=True):
    """
    Filters onshore polygons for a given territory
    (e.g., removing French Guyana from the polygon associated with the French shapefile).

    Parameters
    ----------
    polys : (Multi)Polygon
        Geometry-like object containing the shape of a given onshore region.
    minarea : float
        Area threshold used in the polygon selection process.
    filterremote : boolean

    Returns
    -------
    polys : (Multi)Polygon

    """
    if isinstance(polys, MultiPolygon):

        polys = sorted(polys, key=attrgetter('area'), reverse=True)
        mainpoly = polys[0]
        mainlength = sqrt(mainpoly.area / (2. * pi))

        if mainpoly.area > minarea:

            polys = MultiPolygon([p for p in takewhile(lambda p: p.area > minarea, polys)
                                  if not filterremote or (mainpoly.distance(p) < mainlength)])

        else:

            polys = mainpoly

    return polys


def filter_offshore_polys(offshore_polys, onshore_polys_union, minarea=0.1, filterremote=True):
    """
    Filters offshore polygons for a given territory.

    Parameters
    ----------
    offshore_polys : (Multi)Polygon
        Geometry-like object containing the shape of a given offshore region.
    onshore_polys_union : (Multi)Polygon
        Geometry-like object containing the aggregated shape of given onshore regions.
    minarea : float
        Area threshold used in the polygon selection process.
    filterremote : boolean

    Returns
    -------
    polys : (Multi)Polygon

    """
    if isinstance(offshore_polys, MultiPolygon):

        offshore_polys = sorted(offshore_polys, key=attrgetter('area'), reverse=True)

    else:

        offshore_polys = [offshore_polys]

    mainpoly = offshore_polys[0]
    mainlength = sqrt(mainpoly.area / (5. * pi))
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


def get_onshore_shapes(region_name_list, path_shapefile_data, minarea=0.1, filterremote=True):
    """
    Returns onshore shapefile associated with a given region, or list of regions.

    Parameters
    ----------
    region_name_list : list
        List of regions whose shapefiles are aggregated.
    path_shapefile_data : str
        Relative path of the shapefile data.
    minarea : float
    filterremote : boolean

    Returns
    -------
    onshore_shapes : GeoDataFrame

    """
    filename = 'NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson'
    onshore_shapes = read_file(join(path_shapefile_data, filename))

    onshore_shapes.index = onshore_shapes['CNTR_CODE']

    onshore_shapes = onshore_shapes.reindex(region_name_list)
    onshore_shapes['geometry'] = onshore_shapes['geometry'].map(
        lambda x: filter_onshore_polys(x, minarea, filterremote))

    return onshore_shapes


def get_offshore_shapes(region_name_list, country_shapes, path_shapefile_data, minarea=0.1, filterremote=True):
    """
    Returns offshore shapefile associated with a given region, or list of regions.

    Parameters
    ----------
    region_name_list : list
        List of regions whose shapefiles are aggregated.
    country_shapes : GeoDataFrame
        Dataframe containing onshore shapes of the desired regions.
    path_shapefile_data : str
    minarea : float
    filterremote : boolean

    Returns
    -------
    offshore_shapes : GeoDataFrame

    """
    filename = 'EEZ_RG_01M_2016_4326_LEVL_0.geojson'
    offshore_shapes = read_file(join(path_shapefile_data, filename)).set_index('ISO_ID')

    # Keep only associated countries
    countries_names = [name.split('-')[0] for name in region_name_list]  # Allows to consider states and provinces

    offshore_shapes = offshore_shapes.reindex(countries_names)
    offshore_shapes['geometry'].fillna(Polygon([]), inplace=True)  # Fill nan geometries with empty Polygons

    country_shapes_union = unary_union(country_shapes['geometry'].buffer(0).values)

    # Keep only offshore 'close' to onshore
    offshore_shapes['geometry'] = offshore_shapes['geometry'].map(lambda x: filter_offshore_polys(x,
                                                                                                  country_shapes_union,
                                                                                                  minarea,
                                                                                                  filterremote))

    return offshore_shapes


def chunk_split(l, n):
    """
    Splits large lists in smaller chunks. Done to avoid xarray warnings when slicing large datasets.

    Parameters
    ----------
    l : list
    n : chunk size
    """
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


def xarray_to_ndarray(input_dict):
    """
    Converts dict of xarray objects to ndarray to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    ndarray : ndarray

    """
    key_list = []
    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    array_list = ()

    for region, tech in key_list:
        array_list = (*array_list, input_dict[region][tech].values)

    ndarray = hstack(array_list)

    return ndarray


def xarray_to_dict(input_dict, levels):
    """
    Converts dict of xarray objects to dict of ndarrays to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    levels : int
        Depth of (nested) dict. Available values: 1 or 2.

    Returns
    -------
    output_dict : dict

    """
    output_dict = deepcopy(input_dict)

    if levels == 2:

        key_list = return_dict_keys(input_dict)

        for region, tech in key_list:
            output_dict[region][tech] = input_dict[region][tech].values

    else:

        key_list = input_dict.keys()

        for tech in key_list:
            output_dict[tech] = input_dict[tech].values

    return output_dict


def retrieve_dict_max_length_item(input_dict):
    """
    Retrieve size of largest dict value.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    max_len : int

    """
    key_list = []

    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    max_len = 0

    for region, tech in key_list:

        incumbent_len = len(input_dict[region][tech].locations)

        if incumbent_len > max_len:
            max_len = incumbent_len

    return max_len


def dict_to_xarray(input_dict):
    """
    Converts dict of xarray objects to xarray DataArray to be passed to the optimisation problem.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    dataset : xr.DataArray

    """

    key_list = return_dict_keys(input_dict)

    array_list = []

    for region, tech in key_list:
        array_list.append(input_dict[region][tech])

    dataset = concat(array_list, dim='locations')

    return dataset


def collapse_dict_region_level(input_dict):
    """
    Converts nested dict (dict[region][tech]) to single-level (dict[tech]).

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    output_dict : dict

    """
    output_dict = {}

    technologies = list(set().union(*input_dict.values()))

    for item in technologies:
        l = []
        for region in input_dict:
            for tech in input_dict[region]:
                if tech == item:
                    l.append(input_dict[region][tech])
        output_dict[item] = concat(l, dim='locations')

    return output_dict


def return_dict_keys(input_dict):
    """
    Returns (region, tech) keys of nested dict.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    key_list : list

    """

    key_list = []
    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    return key_list


def concatenate_dict_keys(input_dict):
    """
    Converts nested dict (dict[region][tech]) keys into tuples (dict[(region, tech)]).

    Parameters
    ----------

    Returns
    -------
    output_dict : dict

    """

    output_dict = {}

    key_list = return_dict_keys(input_dict)

    for region, tech in key_list:
        output_dict[(region, tech)] = input_dict[region][tech]

    return output_dict


def retrieve_coordinates_from_dict(input_dict):
    """
    Retrieves coordinate list for each (region, tech) tuple key. Requires dict values to be xarray.DataArray objects!

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    output_dict : dict

    """
    output_dict = {}

    for concat_key in input_dict.keys():
        output_dict[concat_key] = list(input_dict[concat_key].locations.values)

    return output_dict


def compute_generation_potential(capacity_factor_dict, potential_dict):
    """
    Computes generation potential (GWh) to be passed to the optimisation problem.

    Parameters
    ----------
    capacity_factor_dict : dict containing capacity factor time series.

    potential_dict : dict containing technical potential figures per location.

    Returns
    -------
    output_dict : dict

    """
    output_dict = deepcopy(capacity_factor_dict)

    for region in capacity_factor_dict:
        for tech in capacity_factor_dict[region]:
            output_dict[region][tech] = capacity_factor_dict[region][tech] * potential_dict[region][tech]

    return output_dict


def retrieve_tech_coordinates_tuples(input_dict):
    """
    Retrieves list of all (tech, loc) tuples.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    l : list

    """
    l = []

    for key, value in input_dict.items():
        for idx, val in enumerate(value):
            l.append((key, idx))

    return l


def retrieve_incidence_matrix(input_dict):
    """
    Computes the (region, tech) vs. (lon, lat) incidence matrix.

    Parameters
    ----------
    input_dict : dict containing xarray.Dataset objects indexed by (region, tech) tuples.

    Returns
    -------
    incidence_matrix : DataFrame

    """
    coord_list = []
    for concat_key in input_dict.keys():
        coord_list.extend(list(input_dict[concat_key].locations.values))

    idx = list(set(coord_list))
    cols = list(input_dict.keys())

    incidence_matrix = DataFrame(0, index=idx, columns=cols)

    for concat_key in input_dict.keys():

        coordinates = input_dict[concat_key].locations.values

        for c in coordinates:
            incidence_matrix.loc[c, concat_key] = 1

    return incidence_matrix


def retrieve_location_indices_per_tech(input_dict):
    """
    Retrieves integer indices of locations associated with each technology.

    Parameters
    ----------
    input_dict : dict

    Returns
    -------
    output_dict : dict

    """
    key_list = []

    for k1, v1 in input_dict.items():
        for k2, v2 in v1.items():
            key_list.append((k1, k2))

    output_dict = deepcopy(input_dict)

    for region, tech in key_list:
        output_dict[region][tech] = arange(len(input_dict[region][tech].locations))

    return output_dict


def return_region_divisions(region_list, path_shapefile_data):
    # Load countries/regions shapes
    onshore_shapes_all = read_file(join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson'))

    regions = []
    for region in region_list:

        if region == 'EU':
            region_subdivisions = ['AT', 'BE', 'DE', 'DK', 'ES',
                                   'FR', 'UK', 'IE', 'IT', 'LU',
                                   'NL', 'NO', 'PT', 'SE', 'CH', 'CZ',
                                   'AL', 'BG', 'EE', 'LV',
                                   'FI', 'EL', 'HR', 'HU', 'LT',
                                   'PL', 'RO', 'SI', 'SK']
        elif region == 'NA':
            region_subdivisions = ['DZ', 'EG', 'MA', 'LY', 'TN']
        elif region == 'ME':
            region_subdivisions = ['AE', 'BH', 'CY', 'IR', 'IQ', 'IL', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'SY',
                                   'YE']
        elif region == 'US_S':
            region_subdivisions = ['US-TX']
        elif region == 'US_W':
            region_subdivisions = ['US-AZ', 'US-CA', 'US-CO', 'US-MT', 'US-WY', 'US-NM',
                                   'US-UT', 'US-ID', 'US-WA', 'US-NV', 'US-OR']
        elif region == 'US_E':
            region_subdivisions = ['US-ND', 'US-SD', 'US-NE', 'US-KS', 'US-OK', 'US-MN',
                                   'US-IA', 'US-MO', 'US-AR', 'US-LA', 'US-MS', 'US-AL', 'US-TN',
                                   'US-IL', 'US-WI', 'US-MI', 'US-IN', 'US-OH', 'US-KY', 'US-GA', 'US-FL',
                                   'US-PA', 'US-SC', 'US-NC', 'US-VA', 'US-WV',
                                   'US-MD', 'US-DE', 'US-NJ', 'US-NY', 'US-CT', 'US-RI',
                                   'US-MA', 'US-VT', 'US-ME', 'US-NH']
        elif region in onshore_shapes_all['CNTR_CODE'].values:
            region_subdivisions = [region]

        regions.extend(region_subdivisions)

    return regions


def return_region_shapefile(region, path_shapefile_data):
    """
    Returns shapefile associated with the region(s) of interest.

    Parameters
    ----------
    region : str

    path_shapefile_data : str

    Returns
    -------
    output_dict : dict
        Dict object containing i) region subdivisions (if the case) and
        ii) associated onshore and offshore shapes.

    """

    # Load countries/regions shapes
    onshore_shapes_all = read_file(join(path_shapefile_data, 'NUTS_RG_01M_2016_4326_LEVL_0_incl_BA.geojson'))

    if region == 'EU':
        region_subdivisions = ['AT', 'BE', 'DE', 'DK', 'ES',
                               'FR', 'UK', 'IE', 'IT', 'LU',
                               'NL', 'NO', 'PT', 'SE', 'CH', 'CZ',
                               'AL', 'BG', 'EE', 'LV',
                               'FI', 'EL', 'HR', 'HU', 'LT',
                               'PL', 'RO', 'SI', 'SK']
    elif region == 'NA':
        region_subdivisions = ['DZ', 'EG', 'MA', 'LY', 'TN']
    elif region == 'ME':
        region_subdivisions = ['AE', 'BH', 'CY', 'IR', 'IQ', 'IL', 'JO', 'KW', 'LB', 'OM', 'PS', 'QA', 'SA', 'SY', 'YE']
    elif region == 'US_S':
        region_subdivisions = ['US-TX']
    elif region == 'US_W':
        region_subdivisions = ['US-AZ', 'US-CA', 'US-CO', 'US-MT', 'US-WY', 'US-NM',
                               'US-UT', 'US-ID', 'US-WA', 'US-NV', 'US-OR']
    elif region == 'US_E':
        region_subdivisions = ['US-ND', 'US-SD', 'US-NE', 'US-KS', 'US-OK', 'US-MN',
                               'US-IA', 'US-MO', 'US-AR', 'US-LA', 'US-MS', 'US-AL', 'US-TN',
                               'US-IL', 'US-WI', 'US-MI', 'US-IN', 'US-OH', 'US-KY', 'US-GA', 'US-FL',
                               'US-PA', 'US-SC', 'US-NC', 'US-VA', 'US-WV',
                               'US-MD', 'US-DE', 'US-NJ', 'US-NY', 'US-CT', 'US-RI',
                               'US-MA', 'US-VT', 'US-ME', 'US-NH']
    elif region in onshore_shapes_all['CNTR_CODE'].values:
        region_subdivisions = [region]
    else:
        raise ValueError(' Unknown region ', region)

    onshore_shapes_selected = get_onshore_shapes(region_subdivisions, path_shapefile_data)
    offshore_shapes_selected = get_offshore_shapes(region_subdivisions, onshore_shapes_selected, path_shapefile_data)

    onshore = hstack((onshore_shapes_selected["geometry"].values))
    offshore = hstack((offshore_shapes_selected["geometry"].values))

    onshore_union = unary_union(onshore)
    offshore_union = unary_union(offshore)

    onshore_prepared = prepared.prep(onshore_union)
    offshore_prepared = prepared.prep(offshore_union)

    output_dict = {'region_subdivisions': region_subdivisions,
                   'region_shapefiles': {'onshore': onshore_prepared,
                                         'offshore': offshore_prepared}}

    return output_dict


def union_regions(regions, path_shapefile_data, which='both', prepped=True):

    regions = return_region_divisions(regions, path_shapefile_data)

    onshore_shapes_selected = get_onshore_shapes(regions, path_shapefile_data)
    offshore_shapes_selected = get_offshore_shapes(regions, onshore_shapes_selected, path_shapefile_data)

    onshore = hstack((onshore_shapes_selected["geometry"].buffer(0).values))
    offshore = hstack((offshore_shapes_selected["geometry"].buffer(0).values))
    all = hstack((onshore, offshore))

    if which == 'both':
        union = unary_union(all)
    elif which == 'onshore':
        union = unary_union(onshore)
    elif which == 'offshore':
        union = unary_union(offshore)

    if prepped == True:
        full_shape = prepared.prep(union)
    else:
        full_shape = union

    return full_shape


def return_coordinates_from_shapefiles(resource_dataset, shapefiles_region):
    """
    Returning coordinate (lon, lat) pairs falling into the region(s) of interest.

    Parameters
    ----------
    resource_dataset : xarray.Dataset
        Resource dataset.
    shapefiles_region : dict
        Dict object containing the onshore and offshore shapefiles.

    Returns
    -------
    coordinates_in_region : list
        List of coordinate pairs in the region of interest.

    """
    start_coordinates = list(zip(resource_dataset.longitude.values, resource_dataset.latitude.values))

    coordinates_in_region_onshore = array(start_coordinates, dtype('float,float'))[
        [shapefiles_region['onshore'].contains(Point(p)) for p in start_coordinates]].tolist()

    coordinates_in_region_offshore = array(start_coordinates, dtype('float,float'))[
        [shapefiles_region['offshore'].contains(Point(p)) for p in start_coordinates]].tolist()

    coordinates_in_region = list(set(coordinates_in_region_onshore).union(set(coordinates_in_region_offshore)))

    return coordinates_in_region


def return_coordinates_from_shapefiles_light(resource_dataset, shapefiles_region):
    """
    """
    start_coordinates = list(zip(resource_dataset.longitude.values, resource_dataset.latitude.values))

    coordinates_in_region = array(start_coordinates, dtype('float,float'))[
        [shapefiles_region.contains(Point(p)) for p in start_coordinates]].tolist()

    return coordinates_in_region


def retrieve_load_data_partitions(path_load_data, date_slice, alpha, delta, regions, norm_type):
    dict_regions = {'EU': ['AT', 'BE', 'CH', 'DE', 'DK', 'ES',
                           'FR', 'UK', 'IE', 'IT', 'LU',
                           'NL', 'PT', 'SE', 'CZ',
                           'BG', 'CH', 'EE',
                           'FI', 'EL', 'HR', 'HU', 'LT', 'LV', 'PL', 'RO', 'SI', 'SK'],
                    'CWE': ['FR', 'BE', 'LU', 'NL', 'DE'],
                    'BL': ['BE', 'LU', 'NL']}

    load_data = read_csv(join(path_load_data, 'load_opsd_2015_2018.csv'), index_col=0, sep=';')
    load_data.index = date_range('2015-01-01T00:00', '2018-12-31T23:00', freq='H')

    load_data_sliced = load_data.loc[date_slice[0]:date_slice[1]]

    # Adding the stand-alone regions to load dict.
    standalone_regions = list(load_data.columns)
    for region in standalone_regions:
        dict_regions.update({str(region): str(region)})

    if alpha == 'load_central':

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
        load_vector_norm = return_filtered_and_normed(load_vector, delta, norm_type)

    elif alpha == 'load_partition':

        if regions in standalone_regions:
            load_vector = load_data_sliced[dict_regions[regions]]
        else:
            load_vector = load_data_sliced[dict_regions[regions]].sum(axis=1)
        load_vector_norm = return_filtered_and_normed(load_vector, delta, norm_type)

    return load_vector_norm


def return_filtered_and_normed(signal, delta, type='min'):
    l_smooth = signal.rolling(window=delta, center=True).mean().dropna()
    if type == 'min':
        l_norm = (l_smooth - l_smooth.min()) / (l_smooth.max() - l_smooth.min())
    else:
        l_norm = l_smooth / l_smooth.max()

    return l_norm.values


def read_legacy_capacity_data(start_coordinates, region_subdivisions, tech, path_legacy_data):
    """
    Reads dataset of existing RES units in the given area. Available for EU only.

    Parameters
    ----------
    start_coordinates : list
    tech : str
    path_legacy_data : str

    Returns
    -------
    output_dict : dict
        Dict object storing existing capacities per node for a given technology.
    """
    if (tech.split('_')[0] == 'wind') & (tech.split('_')[1] != 'floating'):

        # data = read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127_EL_UK_2010.xls'), sheet_name='Windfarms',
        # header=0, usecols=[2, 5, 9, 10, 18, 22, 23], skiprows=[1])
        data = read_excel(join(path_legacy_data, 'Windfarms_Europe_20200127_EL_UK.xls'), sheet_name='Windfarms',
                          header=0, usecols=[2, 5, 9, 10, 18, 22, 23], skiprows=[1])

        data = data[~data['Latitude'].isin(['#ND'])]
        data = data[~data['Longitude'].isin(['#ND'])]
        data = data[~data['Total power'].isin(['#ND'])]
        data = data[data['Status'] != 'Dismantled']
        data = data[data['ISO code'].isin(region_subdivisions)]
        # data = data[data['Commissioning date'].str.startswith(tuple(['#ND', '201']), na=False)]

        if tech == 'wind_onshore':

            capacity_threshold = 0.2
            data_filtered = data[data['Area'] != 'Offshore'].copy()

        elif tech == 'wind_offshore':

            capacity_threshold = 0.5
            data_filtered = data[data['Area'] == 'Offshore'].copy()

        asset_coordinates = array(list(zip(data_filtered['Longitude'],
                                           data_filtered['Latitude'])))

        node_list = []
        for c in asset_coordinates:
            node_list.append(tuple(start_coordinates[distance.cdist(start_coordinates, [c], 'euclidean').argmin()]))

        data_filtered['Node'] = node_list
        aggregate_capacity_per_node = data_filtered.groupby(['Node'])['Total power'].agg('sum')
        aggregate_capacity_per_node = aggregate_capacity_per_node * (1e-6)

        output_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    elif (tech.split('_')[0] == 'solar') & (tech.split('_')[1] == 'utility'):

        data = read_excel(join(path_legacy_data, 'Solarfarms_Europe_20200208.xlsx'), sheet_name='ProjReg_rpt',
                          header=0, usecols=[0, 3, 4, 5, 8])

        data = data[notnull(data['Coords'])]
        data['Longitude'] = data['Coords'].str.split(',', 1).str[1]
        data['Latitude'] = data['Coords'].str.split(',', 1).str[0]
        data['ISO code'] = data['Country'].map(return_ISO_codes_from_countries())

        data = data[data['ISO code'].isin(region_subdivisions)]

        capacity_threshold = 0.05

        asset_coordinates = array(list(zip(data['Longitude'],
                                           data['Latitude'])))

        node_list = []
        for c in asset_coordinates:
            node_list.append(tuple(start_coordinates[distance.cdist(start_coordinates, [c], 'euclidean').argmin()]))

        data['Node'] = node_list
        aggregate_capacity_per_node = data.groupby(['Node'])['MWac'].agg('sum')
        aggregate_capacity_per_node = aggregate_capacity_per_node * (1e-3)

        output_dict = aggregate_capacity_per_node[aggregate_capacity_per_node > capacity_threshold].to_dict()

    else:

        output_dict = None

    return output_dict


def retrieve_nodes_with_legacy_units(input_dict, region, tech, path_shapefile_data):
    """
    Returns list of nodes where capacity exists.

    Parameters
    ----------
    input_dict : dict
        Dict object storing existing capacities per node for a given technology.
    region : str
        Region.
    tech : str
        Technology.
    path_shapefile_data : str

    Returns
    -------
    existing_locations_filtered : array
        Array populated with coordinate tuples where capacity exists, for a given region and technology.

    """

    if input_dict == None:

        existing_locations_filtered = []

    else:

        existing_locations = list(input_dict.keys())

        if tech in ['wind_offshore', 'wind_floating']:

            # region_shapefile = return_region_shapefile(region, path_shapefile_data)['region_shapefiles']['offshore']
            region_shapefile = union_regions(region, path_shapefile_data, which='offshore')

        else:

            # region_shapefile = return_region_shapefile(region, path_shapefile_data)['region_shapefiles']['onshore']
            region_shapefile = union_regions(region, path_shapefile_data, which='onshore')

        existing_locations_filtered = array(existing_locations, dtype('float,float'))[
            [region_shapefile.contains(Point(p)) for p in existing_locations]].tolist()

    return existing_locations_filtered


def filter_onshore_offshore_locations(coordinates_in_region, spatial_resolution, tech):
    """
    Filters on- and offshore coordinates.

    Parameters
    ----------
    coordinates_in_region : list
    spatial_resolution : float
    tech : str

    Returns
    -------
    updated_coordinates : list
        Coordinates filtered via land/water mask.
    """
    filename = 'ERA5_land_sea_mask_20181231_' + str(spatial_resolution) + '.nc'
    path_land_data = '../input_data/land_data'

    dataset = xr.open_dataset(join(path_land_data, filename))
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])

    dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                 + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))
    array_watermask = dataset['lsm']

    if tech in ['wind_onshore', 'solar_utility', 'solar_residential']:

        mask_watermask = array_watermask.where(array_watermask.data >= 0.2)

    elif tech in ['wind_offshore', 'wind_floating']:

        mask_watermask = array_watermask.where(array_watermask.data < 0.2)

    else:
        raise ValueError(' This technology does not exist.')

    coords_mask_watermask = mask_watermask[mask_watermask.notnull()].locations.values.tolist()

    updated_coordinates = list(set(coordinates_in_region).intersection(set(coords_mask_watermask)))

    return updated_coordinates


def match_point_to_region(point, shape_data, indicator_data):
    """
    Assings a given coordinate tuple (lon, lat) to a NUTS (or any other) region.

    Parameters
    ----------
    point : tuple
        Coordinate in (lon, lat) form.
    shape_data : GeoDataFrame
        Dataframe storing geometries of NUTS regions.
    indicator_data : dict
        Dict object storing technical potential of NUTS regions.

    Returns
    -------
    incumbent_region : str
        Region in which point "p" falls.
    """
    dist = {}

    p = Point(point)

    incumbent_region = None

    for subregion in list(indicator_data.keys()):

        if subregion in shape_data.index:

            if p.within(shape_data.loc[subregion, 'geometry']):
                incumbent_region = subregion

            dist[subregion] = p.distance(shape_data.loc[subregion, 'geometry'])

    if incumbent_region == None:
        print(p, min(dist, key=dist.get))

        incumbent_region = min(dist, key=dist.get)

    return incumbent_region


def return_ISO_codes_from_countries():
    dict_ISO = {'Albania': 'AL', 'Armenia': 'AR', 'Belarus': 'BL', 'Belgium': 'BE', 'Bulgaria': 'BG', 'Croatia': 'HR',
                'Cyprus': 'CY', 'Czech Republic': 'CZ', 'Estonia': 'EE', 'Latvia': 'LV', 'Lithuania': 'LT',
                'Denmark': 'DK', 'France': 'FR', 'Germany': 'DE', 'Greece': 'EL', 'Hungary': 'HU', 'Ireland': 'IE',
                'Italy': 'IT', 'Macedonia': 'MK', 'Malta': 'MT', 'Norway': 'NO', 'Iceland': 'IS', 'Finland': 'FI',
                'Montenegro': 'MN', 'Netherlands': 'NL', 'Poland': 'PL', 'Portugal': 'PT', 'Romania': 'RO',
                'Slovak Republic': 'SK', 'Spain': 'ES', 'Sweden': 'SE',
                'Switzerland': 'CH', 'Turkey': 'TR', 'Ukraine': 'UA', 'United Kingdom': 'UK'}

    return dict_ISO


def get_partition_index(input_dict, deployment_vector, capacity_split='per_tech'):
    """
    Returns start and end indices for each (region, technology) tuple. Required in case the problem
    is defined with partitioning constraints.

    Parameters
    ----------
    input_dict : dict
        Dict object storing coordinates per region and tech.
    deployment_vector : list
        List containing the deployment requirements (un-partitioned or not).
    capacity_split : str
        Capacity splitting rule. To choose between "per_tech" and "per_country".

    Returns
    -------
    index_list : list
        List of indices associated with each (region, technology) tuple.

    """
    key_list = return_dict_keys(input_dict)

    init_index_dict = deepcopy(input_dict)

    regions = list(set([i[0] for i in key_list]))
    technologies = list(set([i[1] for i in key_list]))

    start_index = 0
    for region, tech in key_list:
        init_index_dict[region][tech] = list(arange(start_index, start_index + len(input_dict[region][tech])))
        start_index = start_index + len(input_dict[region][tech])

    if capacity_split == 'per_country':

        if len(deployment_vector) == len(regions):

            index_dict = {key: None for key in regions}
            for region in regions:
                index_list_per_region = []
                tech_list_in_region = [i[1] for i in key_list if i[0] == region]
                for tech in tech_list_in_region:
                    index_list_per_region.extend(init_index_dict[region][tech])
                index_dict[region] = [i for i in index_list_per_region]

        else:

            raise ValueError(' Number of regions ({}) does not match number of deployment constraints ({}).'.format
                             (len(regions), len(deployment_vector)))

    elif capacity_split == 'per_tech':

        if len(deployment_vector) == len(technologies):

            index_dict = {key: None for key in technologies}
            for tech in technologies:
                index_list_per_tech = []
                region_list_with_tech = [i[0] for i in key_list if i[1] == tech]
                for region in region_list_with_tech:
                    index_list_per_tech.extend(init_index_dict[region][tech])
                index_dict[tech] = [i for i in index_list_per_tech]

        else:

            raise ValueError(' Number of technologies ({}) does not match number of deployment constraints ({}).'.format
                             (len(technologies), len(deployment_vector)))

    elif capacity_split == 'per_country_and_tech':

        index_dict = init_index_dict

    for region, tech in key_list:
        index_dict[region][tech] = [i + 1 for i in index_dict[region][tech]]

    return index_dict


def read_inputs(inputs):
    """

    Parameters
    ----------
    inputs :

    Returns
    -------

    """
    with open(inputs) as infile:
        data = yaml.safe_load(infile)

    return data


def init_folder(parameters, input_dict, suffix=None):
    """Initilize an output folder.

    Parameters:

    ------------

    parameters : dict
        Parameters dictionary.

    Returns:

    ------------

    path : str
        Relative path of the folder.


    """
    prefix = str(parameters['name_prefix'])
    no_locs = sum(parameters['deployment_dict'].values())
    no_part = len(parameters['deployment_dict'])
    total_locs = input_dict['criticality_data'].shape[1]
    no_yrs = round((to_datetime(parameters['time_slice'][1]) - to_datetime(parameters['time_slice'][0])) / timedelta64(1, 'Y'), 0)

    if not isdir("../output_data"):
        makedirs(abspath("../output_data"))

        path = abspath('../output_data/' + prefix + total_locs + 'locs_' + no_yrs + 'y_n' + no_locs + 'k_' + no_part + suffix)
        makedirs(path)

    else:
        path = abspath('../output_data/' + prefix + total_locs + 'locs_' + no_yrs + 'y_n' + no_locs + 'k_' + no_part + suffix)
        makedirs(path)

    custom_log(' Folder path is: {}'.format(str(path)))

    if parameters['keep_files'] == False:
        custom_log(' WARNING! Files will be deleted at the end of the run.')

    return path


def generate_jl_output(deployment_dict, criticality_matrix, filtered_coordinates):

    concat_deployment_dict = concatenate_dict_keys(deployment_dict)
    region_list = [tuple for tuple in concat_deployment_dict.keys()]

    int_to_region_map = {}
    for idx, region in enumerate(region_list):
        int_to_region_map[region] = idx + 1

    deployment_dict_int = dict(zip(int_to_region_map.values(), concat_deployment_dict.values()))

    index_dict = concatenate_dict_keys(get_partition_index(filtered_coordinates, deployment_dict,
                                                           capacity_split='per_country_and_tech'))
    index_dict_swap = {k: oldk for oldk, oldv in index_dict.items() for k in oldv}
    for key, value in index_dict_swap.items():
        index_dict_swap[key] = int_to_region_map[value]

    output_dict = {'deployment_dict': deployment_dict_int,
                   'criticality_matrix': criticality_matrix,
                   'index_dict': index_dict_swap}

    return output_dict


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


def custom_log(message):
    """
    Parameters
    ----------
    message : str

    """
    print(datetime.now().strftime('%H:%M:%S') + ' --- ' + str(message))


import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi, arccos


def distsphere(lat1, long1, lat2, long2):
    """Calculates distance between two points on a sphere.

    Parameters:

    ------------

    lat1, lon1, lat2, lon2 : float
        Geographical coordinates of the two points.




    Returns:

    ------------

   arc : float
        Distance between points in radians.

    """

    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.
    cosine = (sin(phi1) * sin(phi2) * cos(theta1 - theta2) + cos(phi1) * cos(phi2))
    arc = arccos(cosine)

    # Remember to multiply arc by the radius of the earth!
    return arc


def update_latitude(lat1, arc):
    """Helper function that adjusts the central latitude position.

    Parameters:

    ------------

    lat1 : float

    arc : float




    Returns:

    ------------

   lat2 : float

    """

    degrees_to_radians = pi / 180.0
    lat2 = (arc - ((90 - lat1) * degrees_to_radians)) * (1. / degrees_to_radians) + 90
    return lat2


def centerMap(lons, lats):
    """Returns elements of the Basemap plot (center latitude and longitude,
    height and width of the map).

    Parameters:

    ------------

    lons : list

    lats : list



    Returns:

    ------------

    lon0, lat0, mapW, mapH : float

    """
    # Assumes -90 < Lat < 90 and -180 < Lon < 180, and
    # latitude and logitude are in decimal degrees
    earthRadius = 6378100.0  # earth's radius in meters

    lon0 = ((max(lons) - min(lons)) / 2) + min(lons)

    b = distsphere(max(lats), min(lons), max(lats), max(lons)) * earthRadius / 2
    c = distsphere(max(lats), min(lons), min(lats), lon0) * earthRadius

    # use pythagorean theorom to determine height of plot
    mapH = np.sqrt(c ** 2 - b ** 2)
    mapW = distsphere(min(lats), min(lons), min(lats), max(lons)) * earthRadius

    arcCenter = (mapH / 2) / earthRadius
    lat0 = update_latitude(min(lats), arcCenter)

    minlon = min(lons) - 1
    maxlon = max(lons) + 1
    minlat = min(lats) - 1
    maxlat = max(lats) + 1

    return lon0, lat0, minlon, maxlon, minlat, maxlat, mapH, mapW


def plot_basemap(coordinate_list, title):
    longitudes = [i[0] for i in coordinate_list]
    latitudes = [i[1] for i in coordinate_list]

    lon0, lat0, minlon, maxlon, minlat, maxlat, mapH, mapW = centerMap(longitudes, latitudes)

    land_50m = cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                            edgecolor='darkgrey',
                                            facecolor=cfeature.COLORS['land_alt1'])

    proj = ccrs.PlateCarree()
    plt.figure(figsize=(10, 6))

    ax = plt.axes(projection=proj)
    ax.set_extent([minlon, maxlon, minlat, maxlat], proj)

    ax.add_feature(land_50m, linewidth=0.5)
    ax.add_feature(cfeature.OCEAN, facecolor='white')
    ax.add_feature(cfeature.LAKES, facecolor='white')
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5)

    ax.scatter(longitudes, latitudes, transform=proj, marker='x', color='red', s=mapW / 1e6, zorder=10, alpha=1.0)
    ax.set_extent([-15., 45., 30., 75.], crs=proj)
    ax.set_title(title)

    plt.show()
