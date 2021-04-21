from copy import deepcopy
from os import makedirs
from os.path import join, isdir, abspath
import warnings
import pycountry as pyc
import xarray as xr
import yaml
from geopandas import read_file, GeoSeries
from numpy import hstack, arange, dtype, array, timedelta64, nan, sum, deg2rad, sin, cos, arccos, ceil, int64
from pandas import read_csv, to_datetime, Series, notnull, MultiIndex
from shapely import prepared
from shapely.geometry import Point
from shapely.ops import unary_union
from xarray import concat
import geopy

import logging
logging.basicConfig(level=logging.INFO, format=f"%(levelname)s %(asctime)s - %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)


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

    key_list = return_dict_keys(input_dict)

    array_list = []
    for region, tech in key_list:
        array_list.append(input_dict[region][tech])
    dataset = xr.concat(array_list, dim='locations')
    ndarray = dataset.values

    return ndarray


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


def get_deployment_vector(regions, technologies, deployments):

    d = {}
    for i, region in enumerate(regions):
        d[region] = {}
        for j, tech in enumerate(technologies):
            d[region][tech] = deployments[i][j]

    return d


def return_region_divisions(region_list, data_path):

    arcgis_fn = f"{data_path}input/shapefiles/Longitude_Graticules_and_World_Countries_Boundaries-shp/" \
                f"99bfd9e7-bb42-4728-87b5-07f8c8ac631c2020328-1-1vef4ev.lu5nk.shp"
    shapes = read_file(arcgis_fn)
    shapes["CNTRY_NAME"] = shapes["CNTRY_NAME"].apply(convert_old_country_names)
    shapes["iso2"] = Series(convert_country_codes(shapes["CNTRY_NAME"].values, "name", "alpha_2"))
    shapes = shapes[notnull(shapes["iso2"])]
    shapes = shapes.set_index("iso2")['geometry']

    regions = []
    for region in region_list:

        if region == 'EU':
            region_subdivisions = ['AT', 'BE', 'DE', 'DK', 'ES',
                                   'FR', 'GB', 'IE', 'IT', 'LU',
                                   'NL', 'NO', 'PT', 'SE', 'CH', 'CZ',
                                   'EE', 'LV', 'RO', 'BG', 'HR', 'RS',
                                   'FI', 'GR', 'HR', 'HU', 'LT',
                                   'PL', 'SI', 'SK']
        elif region == 'NA':
            region_subdivisions = ['DZ', 'EG', 'MA', 'LY', 'TN']
        elif region == 'ME':
            region_subdivisions = ['AE', 'BH', 'CY', 'IR', 'IQ', 'IL', 'JO', 'KW', 'LB', 'OM',
                                   'PS', 'QA', 'SA', 'SY', 'YE']
        elif region == 'CWE':
            region_subdivisions = ['FR', 'BE', 'LU', 'NL', 'DE']
        elif region in shapes.index:
            region_subdivisions = [region]
        else:
            warnings.warn(f"{region} not in shapes list!")
            continue

        regions.extend(region_subdivisions)

    return regions


def convert_old_country_names(c):
    """Converting country old full names to new ones, as some datasets are not updated on the issue."""

    if c == "Macedonia":
        return "North Macedonia"

    if c == "Czech Republic":
        return "Czechia"

    if c == 'Syria':
        return 'Syrian Arab Republic'

    if c == 'Iran':
        return 'Iran, Islamic Republic of'

    if c == "Byelarus":
        return "Belarus"

    return c


def remove_landlocked_countries(country_list):
    """Filtering out landlocked countries."""
    landlocked_countries = {'AT', 'BY', 'CH', 'CZ', 'HU', 'LI', 'LU', 'MD', 'MK', 'RS', 'SK'}
    return sorted(list(set(country_list) - landlocked_countries))


def convert_country_codes(source_codes, source_format, target_format, throw_error = False):
    """
    Convert country codes, e.g., from ISO_2 to full name.

    Parameters
    ----------
    source_codes: List[str]
        List of codes to convert.
    source_format: str
        Format of the source codes (alpha_2, alpha_3, name, ...)
    target_format: str
        Format to which code must be converted (alpha_2, alpha_3, name, ...)
    throw_error: bool (default: False)
        Whether to throw an error if an attribute does not exist.

    Returns
    -------
    target_codes: List[str]
        List of converted codes.
    """
    target_codes = []
    for code in source_codes:
        try:
            country_codes = pyc.countries.get(**{source_format: code})
            if country_codes is None:
                raise KeyError(f"Data is not available for code {code} of type {source_format}.")
            target_code = getattr(country_codes, target_format)
        except (KeyError, AttributeError) as e:
            if throw_error:
                raise e
            target_code = nan
        target_codes += [target_code]
    return target_codes


def get_onshore_shapes(regions, data_path):
    """
    Return onshore shapes from naturalearth data (ISO_2 codes).

    """

    arcgis_fn = f"{data_path}input/shapefiles/Longitude_Graticules_and_World_Countries_Boundaries-shp/" \
                f"99bfd9e7-bb42-4728-87b5-07f8c8ac631c2020328-1-1vef4ev.lu5nk.shp"
    shapes = read_file(arcgis_fn)
    shapes["CNTRY_NAME"] = shapes["CNTRY_NAME"].apply(convert_old_country_names)
    shapes["iso2"] = Series(convert_country_codes(shapes["CNTRY_NAME"].values, "name", "alpha_2"))
    shapes = shapes[notnull(shapes["iso2"])]
    shapes = shapes.set_index("iso2")['geometry']

    if regions is not None:
        missing_codes = set(regions) - set(shapes.index)
        assert not missing_codes, f"Shapes are not available for the " \
                                  f"following codes: {sorted(list(missing_codes))}"
        shapes = shapes[regions]

    return shapes


def get_offshore_shapes(regions, data_path):
    """
    Return offshore shapes for a list of regions.
    """

    # Remove landlocked countries for which there is no offshore shapes
    iso_codes = remove_landlocked_countries(regions)

    eez_fn = f"{data_path}input/shapefiles/eez/World_EEZ_v8_2014.shp"
    eez_shapes = read_file(eez_fn)

    eez_shapes = eez_shapes[notnull(eez_shapes['ISO_3digit'])]
    # Create column with ISO_A2 code.
    eez_shapes['ISO_A2'] = convert_country_codes(eez_shapes['ISO_3digit'].values, 'alpha_3', 'alpha_2')
    eez_shapes = eez_shapes[["geometry", "ISO_A2"]].dropna()
    eez_shapes = eez_shapes.set_index('ISO_A2')["geometry"]

    # Filter shapes
    missing_codes = set(iso_codes) - set(eez_shapes.index)
    assert not missing_codes, f"Error: No shapes available for codes {sorted(list(missing_codes))}"
    eez_shapes = eez_shapes[iso_codes]

    # Combine polygons corresponding to the same countries.
    unique_codes = set(eez_shapes.index)
    offshore_shapes = GeoSeries(name='geometry')
    for c in unique_codes:
        # TODO: quick hack to retrieve the continental waters for ES, PT. To be properly addressed.
        if c == 'ES':
            # offshore_shapes[c] = unary_union(eez_shapes[c])
            offshore_shapes[c] = eez_shapes.loc[c].iloc[1]
        elif c == 'PT':
            offshore_shapes[c] = eez_shapes.loc[c].iloc[2]
        else:
            offshore_shapes[c] = eez_shapes[c]

    return offshore_shapes


def union_regions(regions, data_path, which='both', prepped=True):

    regions = return_region_divisions(regions, data_path)

    if which == 'both':
        onshore_shapes_selected = get_onshore_shapes(regions, data_path)
        offshore_shapes_selected = get_offshore_shapes(regions, data_path)

        onshore = hstack(onshore_shapes_selected.values)
        offshore = hstack(offshore_shapes_selected.values)
        all_shapes = hstack((onshore, offshore))

        union = unary_union(all_shapes)

    elif which == 'onshore':
        onshore_shapes_selected = get_onshore_shapes(regions, data_path)
        onshore = hstack(onshore_shapes_selected.values)

        union = unary_union(onshore)

    elif which == 'offshore':
        offshore_shapes_selected = get_offshore_shapes(regions, data_path)
        offshore = hstack(offshore_shapes_selected.values)

        union = unary_union(offshore)

    if prepped:
        full_shape = prepared.prep(union)
    else:
        full_shape = union

    return full_shape


def return_coordinates_from_shapefiles(resource_dataset, shapefiles_region):
    """
    """
    start_coordinates = list(zip(resource_dataset.longitude.values, resource_dataset.latitude.values))

    coordinates_in_region = array(start_coordinates, dtype('float,float'))[
        [shapefiles_region.contains(Point(p)) for p in start_coordinates]].tolist()

    return coordinates_in_region


def capacity_to_cardinality(dataset, model_config, tech_config, site_coordinates_dict, legacy_coordinates_dict,
                            reference_lat=55, reference_lon=10, lat_dist_per_deg=111):

    spatial_resolution = model_config['spatial_resolution']
    deployment_dict = get_deployment_vector(model_config['regions'],
                                            model_config['technologies'],
                                            model_config['deployments'])

    lon1 = deg2rad(reference_lon)
    lon2 = deg2rad(reference_lon+spatial_resolution)
    reference_lat = deg2rad(reference_lat)

    dist_lat = lat_dist_per_deg*spatial_resolution
    dist_lon = arccos(sin(reference_lat)*sin(reference_lat)+cos(reference_lat)*cos(reference_lat)*cos(lon2-lon1))*6371
    cell_area = dist_lat*dist_lon

    cardinality_dict = deepcopy(deployment_dict)
    adj_cardinality_dict = deepcopy(deployment_dict)
    legacy_dict = deepcopy(deployment_dict)
    key_list = return_dict_keys(deployment_dict)

    for region, tech in key_list:

        tech_potential = tech_config[tech]['power_density'] * tech_config[tech]['land_utilization_factor'] * cell_area
        cardinality_dict[region][tech] = int(ceil(deployment_dict[region][tech]/tech_potential*1e3))

        shape_region = union_regions([region], model_config['data_path'], which=tech_config[tech]['where'])
        points_in_region = return_coordinates_from_shapefiles(dataset, shape_region)
        legacy_dict[region][tech] = list(set(legacy_coordinates_dict[tech]).intersection(set(points_in_region)))
        adj_cardinality_dict[region][tech] = min(len(site_coordinates_dict[region][tech]),
                                                 max(len(legacy_dict[region][tech]), cardinality_dict[region][tech]))

        logger.info(f"Picking {adj_cardinality_dict[region][tech]} {tech} sites in {region} out of "
                    f"{len(site_coordinates_dict[region][tech])} candidate ones.")

    return adj_cardinality_dict


def get_potential_per_site(input_dict, tech_parameters, spatial_resolution):
    """Compute cell potential of candidate siting locations based on reanalysis grids with different resolutions.
     Parameters
     ----------
     input_dict: dict
     tech_parameters: dict
     spatial_resolution: float
     Returns
     -------
     output_dict: dict
     """

    key_list = return_dict_keys(input_dict)
    output_dict = deepcopy(input_dict)

    for region, tech in key_list:
        tech_dict = tech_parameters[tech]
        locations = MultiIndex.from_tuples(input_dict[region][tech].locations.values, names=('longitude', 'latitude'))
        potentials = Series(0., index=locations)

        for (lon, lat) in potentials.index:
            lat_south = (lon, lat - spatial_resolution / 2.)
            lat_north = (lon, lat + spatial_resolution / 2.)
            lon_west = (lon - spatial_resolution / 2., lat)
            lon_east = (lon + spatial_resolution / 2., lat)

            dist_lat = geopy.distance.distance(geopy.distance.lonlat(*lat_south), geopy.distance.lonlat(*lat_north)).km
            dist_lon = geopy.distance.distance(geopy.distance.lonlat(*lon_west), geopy.distance.lonlat(*lon_east)).km
            # 1e-3 converts from MW to GW
            potentials[(lon, lat)] = \
                dist_lat * dist_lon * tech_dict['power_density'] * tech_dict['land_utilization_factor'] * 1e-3

        output_dict[region][tech] = xr.DataArray.from_series(potentials).stack(locations=('longitude', 'latitude'))\
            .dropna(dim='locations').reindex_like(input_dict[region][tech])

    return output_dict


def smooth_load_data(data_path, regions, date_slice, delta):

    load_data_fn = join(data_path, 'input/load_data', 'load_entsoe_2006_2020_full.csv')
    load_data = read_csv(load_data_fn, index_col=0)
    load_data.index = to_datetime(load_data.index)
    load_data_sliced = load_data.loc[date_slice[0]:date_slice[1]]

    regions_list = return_region_divisions(regions, data_path)
    load_vector = load_data_sliced[regions_list]

    load_vector_rolling = load_vector.rolling(window=delta, center=True).mean().dropna()

    return load_vector_rolling


def norm_load_by_load(load_vector_rolling, norm):

    if norm == 'min':
        load_vector_norm = (load_vector_rolling - load_vector_rolling.min()) / \
                           (load_vector_rolling.max() - load_vector_rolling.min())
    else:
        load_vector_norm = load_vector_rolling.divide(load_vector_rolling.max())

    return load_vector_norm


def norm_load_by_deployments(load_vector_rolling, deployments):

    load_vector_norm = load_vector_rolling.divide(deployments)

    return load_vector_norm


def filter_onshore_offshore_locations(coordinates_in_region, data_path, spatial_resolution, tech_dict, tech):
    """
    Filters on- and offshore coordinates.

    Parameters
    ----------
    coordinates_in_region : list
    data_path: str
    spatial_resolution : float
    tech_dict: dict
    tech : str

    Returns
    -------
    updated_coordinates : list
        Coordinates filtered via land/water mask.
    """

    assert tech in ['wind_onshore', 'pv_utility', 'pv_residential', 'wind_offshore', 'wind_floating'], \
        f"Technology {tech} not available."

    land_fn = 'ERA5_surface_characteristics_20181231_' + str(spatial_resolution) + '.nc'
    land_path = join(data_path, 'input/land_data', land_fn)

    dataset = xr.open_dataset(land_path)
    dataset = dataset.sortby([dataset.longitude, dataset.latitude])
    dataset = dataset.assign_coords(longitude=(((dataset.longitude
                                                 + 180) % 360) - 180)).sortby('longitude')
    dataset = dataset.drop('time').squeeze().stack(locations=('longitude', 'latitude'))

    depth_threshold_low = tech_dict['depth_threshold_low']
    depth_threshold_high = tech_dict['depth_threshold_high']

    array_watermask = dataset['lsm']
    # Careful with this one because max depth is 999.
    array_bathymetry = dataset['wmb'].fillna(0.)

    mask_offshore = array_bathymetry.where(((array_bathymetry.data < depth_threshold_low) |
                                            (array_bathymetry.data > depth_threshold_high)) |
                                           (array_watermask.data > 0.1))
    coords_mask_offshore = mask_offshore[mask_offshore.notnull()].locations.values.tolist()

    if tech in ['wind_onshore', 'pv_utility', 'pv_residential']:
        updated_coordinates = list(set(coordinates_in_region).intersection(set(coords_mask_offshore)))

    elif tech in ['wind_offshore', 'wind_floating']:
        updated_coordinates = list(set(coordinates_in_region).difference(set(coords_mask_offshore)))

    return updated_coordinates


def get_partition_index(input_dict):
    """
    Returns start and end indices for each (region, technology) tuple. Required in case the problem
    is defined with partitioning constraints.

    Parameters
    ----------
    input_dict : dict
        Dict object storing coordinates per region and tech.

    Returns
    -------
    index_dict : dict
        Dict of indices associated with each (region, technology) tuple.

    """
    key_list = return_dict_keys(input_dict)
    index_dict = deepcopy(input_dict)

    start_index = 0
    for region, tech in key_list:
        indices = list(arange(start_index, start_index + len(input_dict[region][tech])))
        index_dict[region][tech] = [i + 1 for i in indices]
        start_index = start_index + len(input_dict[region][tech])

    return index_dict


def init_folder(parameters, no_locs, c, suffix=None):
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
    output_data_path = join(parameters['data_path'], 'output')

    no_locs = str(no_locs)
    no_part = str(len(parameters['regions']))
    no_yrs = str(int(round((to_datetime(parameters['time_slice'][1]) -
                            to_datetime(parameters['time_slice'][0])) / timedelta64(1, 'Y'), 0)))
    c = str(c)

    if not isdir(output_data_path):
        makedirs(abspath(output_data_path))

    path = abspath(output_data_path + '/' + no_yrs + 'y_n' + no_locs + '_k' + no_part + '_c' + c + suffix)
    makedirs(path)

    return path


def generate_jl_input(deployment_dict, filtered_coordinates, site_positions, legacy_sites):

    concat_deployment_dict = concatenate_dict_keys(deployment_dict)
    region_list = [tuple for tuple in concat_deployment_dict.keys()]

    int_to_region_map = {}
    for idx, region in enumerate(region_list):
        int_to_region_map[region] = idx + 1

    legacy_sites_index = []

    for tech in legacy_sites:
        if len(legacy_sites[tech]) > 0:
            for site in legacy_sites[tech]:
                tech_site_tuple = (tech, site)
                for s in site_positions:
                    if site_positions[s] == tech_site_tuple:
                        legacy_sites_index.append(s+1)
        else:
            continue

    deployment_dict_int = dict(zip(int_to_region_map.values(), concat_deployment_dict.values()))

    index_dict = concatenate_dict_keys(get_partition_index(filtered_coordinates))
    index_dict_swap = {k: oldk for oldk, oldv in index_dict.items() for k in oldv}
    for key, value in index_dict_swap.items():
        index_dict_swap[key] = int_to_region_map[value]

    output_dict = {'deployment_dict': {int64(k): int64(v) for k, v in deployment_dict_int.items()},
                   'index_dict': {int64(k): int64(v) for k, v in index_dict_swap.items()},
                   'legacy_site_list': array(sorted(legacy_sites_index)).astype('int64')}

    return output_dict


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
