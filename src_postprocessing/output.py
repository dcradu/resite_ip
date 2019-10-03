import matplotlib.pyplot as plt
import pandas as pd
from os.path import join, abspath
from src.tools import read_database, get_global_coordinates, return_output, return_coordinates_from_countries, selected_data, read_load_data, filter_offshore_coordinates
from output_tools import read_output, read_inputs_plotting, plot_basemap
from itertools import chain, combinations, cycle, islice
from collections import defaultdict
from random import randint
from copy import deepcopy
from numpy import array, where, tile, quantile, ceil, nonzero, split, diff, sum, arange, histogram, cumsum, max, searchsorted, append
from xarray import concat
import seaborn as sns
from matplotlib.lines import Line2D
import numpy as np


class Output(object):


    def __init__(self, path):

        self.output_folder_path = abspath(join('../output_data/', path))
        self.parameters_yaml = read_inputs_plotting(self.output_folder_path)
        self.outputs_pickle = read_output(path)

    def return_numerics(self, choice, ndiff=1, firm_threshold=0.3):
        """Function returning statistical measures associatd with the aggregation
        of different resource time series.

            Parameters:

            ------------

            choice : list
                The time series aggregation to assess.
                    "opt" - the (optimal) selection associated with the suggested siting problem
                    "max" - the selection of n locations maximizing production
                    "rand" - the selection of n random sites

            ndiff : int
                Defines the nth difference to be computed within time series.

            firm_threshold : float
                Defines the threshold that defines the "firmness" of time series

            Returns:

            ------------

            result_dict : dict
                Dictionary containing various indexed indicators to be plotted.
        """

        print('Problem: {}'.format(self.parameters_yaml['main_problem'], end=', '))
        print('Objective: {}'.format(self.parameters_yaml['main_objective'], end=', '))
        print('Spatial resolution: {}'.format(self.parameters_yaml['spatial_resolution'], end=', '))
        print('Time horizon: {}'.format(self.parameters_yaml['time_slice'], end=', '))
        print('Measure: {}'.format(self.parameters_yaml['resource_quality_measure'], end=', '))
        print('alpha: {}'.format(self.parameters_yaml['alpha_rule'], end=', '))
        print('delta: {}'.format(self.parameters_yaml['delta'], end=', '))
        print('beta: {}'.format(self.parameters_yaml['beta']))

        path_resource_data = self.parameters_yaml['path_resource_data'] + str(self.parameters_yaml['spatial_resolution']) + '/'
        horizon = self.parameters_yaml['time_slice']
        technologies = self.parameters_yaml['technologies']
        path_transfer_function_data = self.parameters_yaml['path_transfer_function_data']
        path_load_data = self.parameters_yaml['path_load_data']
        regions = self.parameters_yaml['regions']
        no_sites = self.parameters_yaml['cardinality_constraint']

        price_ts = pd.read_csv('../input_data/el_price/elix_2014_2018.csv', index_col=0, sep=';')
        price_ts = price_ts['price']
        length_timeseries = self.outputs_pickle['capacity_factors_dict'][technologies[0]].shape[0]
        threshold_depth = self.parameters_yaml['depth_threshold']
        spatial_resolution = self.parameters_yaml['spatial_resolution']
        path_landseamask = self.parameters_yaml['path_landseamask']

        if length_timeseries <= price_ts.size:
            price_ts = price_ts[:length_timeseries]
        else:
            el_ts_multiplier = int(ceil(length_timeseries / price_ts.size))
            price_ts = tile(price_ts, el_ts_multiplier)
            price_ts = price_ts[:length_timeseries]

        load_dict, load_data = read_load_data(path_load_data, horizon)

        regions_df = []
        for item in regions:
            regions_df.extend(load_dict[item])

        load_ts = load_data[regions_df].sum(axis=1)

        signal_dict = dict.fromkeys(choice, None)
        firmness_dict = dict.fromkeys(choice, None)
        difference_dict = dict.fromkeys(choice, None)
        result_dict = dict.fromkeys(['signal', 'difference', 'firmness'], None)


        database = read_database(path_resource_data)
        global_coordinates = get_global_coordinates(database, self.parameters_yaml['spatial_resolution'],
                                                    self.parameters_yaml['population_density_threshold'],
                                                    self.parameters_yaml['path_population_density_data'],
                                                    self.parameters_yaml['protected_areas_selection'],
                                                    self.parameters_yaml['protected_areas_threshold'],
                                                    self.parameters_yaml['path_protected_areas_data'],
                                                    population_density_layer=False, protected_areas_layer=False)
        coordinates_filtered_depth = filter_offshore_coordinates(global_coordinates,
                                                                    threshold_depth,
                                                                    spatial_resolution,
                                                                    path_landseamask)



        for c in choice:

            if c == 'COMP':

                array_list = []
                for tech in self.outputs_pickle['optimal_location_dict'].keys():
                    array_per_tech = array(self.outputs_pickle['capacity_factors_dict'][tech].sel(
                        locations=self.outputs_pickle['optimal_location_dict'][tech]).values).sum(axis=1)
                    array_list.append(array_per_tech)

            elif c == 'RAND':

                region_coordinates = return_coordinates_from_countries(regions, coordinates_filtered_depth, add_offshore=True)
                truncated_data = selected_data(database, region_coordinates, horizon)
                output_data = return_output(truncated_data, technologies, path_transfer_function_data)

                no_coordinates = sum(fromiter((len(lst) for lst in region_coordinates.values()), dtype=int))

                output = []
                for item in output_data.keys():
                    output.append(output_data[item])

                output_overall = concat(output, dim='locations')

                score_init = 0.

                for i in range(500):

                    location_list = []
                    ts_list = []

                    idx = [randint(0, no_coordinates*len(technologies) - 1) for x in range(sum(no_sites))]
                    for loc in idx:
                        location_list.append(output_overall.isel(locations=loc).locations.values.flatten()[0])
                        ts_list.append(output_overall.isel(locations=loc).values)

                    score = array(ts_list).sum()

                    if score > score_init:
                        score_init = score
                        ts_incumbent = ts_list

                array_list = ts_incumbent

            elif c == 'PROD':

                suboptimal_dict = dict.fromkeys(self.parameters_yaml['regions'], None)
                suboptimal_dict_ts = deepcopy(suboptimal_dict)

                if len(no_sites) == 1:

                    location_list = []
                    ts_list = []
                    truncated_data_list = []
                    output_data_list = []

                    for key in suboptimal_dict.keys():

                        region_coordinates = return_coordinates_from_countries(key, coordinates_filtered_depth, add_offshore=True)
                        truncated_data = selected_data(database, region_coordinates, horizon)

                        for k in technologies:

                            tech = []
                            tech.append(k)

                            output_data = return_output(truncated_data, tech, path_transfer_function_data)[k]
                            output_data_list.append(output_data)

                            truncated_data_sum = output_data.sum(dim='time')
                            truncated_data_list.append(truncated_data_sum)

                    truncated_data_concat = concat(truncated_data_list, dim='locations')
                    output_data_concat = concat(output_data_list, dim='locations')

                    tdata = truncated_data_concat.argsort()[-no_sites[0]:]

                    for loc in tdata.values:
                        location_list.append(output_data_concat.isel(locations=loc).locations.values.flatten()[0])
                        ts_list.append(output_data_concat.isel(locations=loc).values)

                    array_list = ts_list

                else:

                    idx = 0

                    for key in suboptimal_dict.keys():


                        location_list = []
                        ts_list = []
                        output_data_list = []
                        truncated_data_list_per_region = []

                        region_coordinates = return_coordinates_from_countries(key, global_coordinates)
                        truncated_data = selected_data(database, region_coordinates, horizon)

                        for k in technologies:

                            tech = []
                            tech.append(k)

                            output_data = return_output(truncated_data, tech, path_transfer_function_data)[k]
                            output_data_list.append(output_data)

                            truncated_data_sum = output_data.sum(dim='time')
                            truncated_data_list_per_region.append(truncated_data_sum)

                        truncated_data_concat_per_region = concat(truncated_data_list_per_region, dim='locations')
                        output_data_concat = concat(output_data_list, dim='locations')

                        tdata = truncated_data_concat_per_region.argsort()[-no_sites[idx]:]

                        for loc in tdata.values:
                            location_list.append(output_data_concat.isel(locations=loc).locations.values.flatten()[0])
                            ts_list.append(output_data_concat.isel(locations=loc).values)

                        idx += 1

                        suboptimal_dict[key] = location_list
                        suboptimal_dict_ts[key] = ts_list

                    array_list = []
                    for region in suboptimal_dict_ts.keys():
                        array_per_tech = array(suboptimal_dict_ts[region]).sum(axis=0)
                        array_list.append(array_per_tech)

            elif c == 'NSEA':

                location_list = []
                ts_list = []

                region_coordinates = return_coordinates('NSea', coordinates_filtered_depth)
                truncated_data = selected_data(database, region_coordinates, horizon)
                output_data = return_output(truncated_data, technologies, path_transfer_function_data)

                truncated_data_sum = output_data['wind_aerodyn'].sum(dim='time')

                tdata = truncated_data_sum.argsort()[-no_sites[0]:]

                for loc in tdata.values:
                    location_list.append(output_data['wind_aerodyn'].isel(locations=loc).locations.values.flatten()[0])
                    ts_list.append(output_data['wind_aerodyn'].isel(locations=loc).values)

                array_list = ts_list

            array_sum = pd.Series(data=array(array_list).sum(axis=0))
            difference = array_sum.diff(periods=ndiff).dropna()
            firmness = assess_firmness(array_sum, firm_threshold * sum(self.parameters_yaml['cardinality_constraint']))




            print('-------------------------------------')
            print('NUMERICAL RESULTS FOR THE {} SET OF SITES.'.format(str(c)))

            print('Variance of time series: {}'.format(round(array_sum.var(), 4)))
            print('Mean of time series: {}'.format(round(array_sum.mean(), 4)))
            print('Mean +/- std of time series: {}'.format(round(array_sum.mean() - array_sum.std(), 4)))
            print('p1, p5, p10 of time series: {}, {}, {}'.format(round(array_sum.quantile(q=0.01), 4),
                                                                  round(array_sum.quantile(q=0.05), 4),
                                                                  round(array_sum.quantile(q=0.1), 4)))
            print('{} difference count within +/- 1%, 5% of total output: {}, {}'.format(ndiff, difference.between(
                left=-0.01 * sum(self.parameters_yaml['cardinality_constraint']),
                right=0.01 * sum(self.parameters_yaml['cardinality_constraint'])).sum(), difference.between(
                left=-0.05 * sum(self.parameters_yaml['cardinality_constraint']),
                right=0.05 * sum(self.parameters_yaml['cardinality_constraint'])).sum()))
            print('Total estimated revenue: {}'.format(round(price_ts * array_sum).sum(), 4))
            print('Estimated low-yield (10, 20, 30%) revenue : {}, {}, {}'.format(
                round(clip_revenue(array_sum, price_ts, 0.1), 4),
                round(clip_revenue(array_sum, price_ts, 0.2), 4),
                round(clip_revenue(array_sum, price_ts, 0.3), 4)))
            print('Estimated capacity credit for top (1, 5, 10%) peak demand hours (valid for centralized planning only): {}, {}, {}'.format(
                round(assess_capacity_credit(load_ts, array_sum, sum(self.parameters_yaml['cardinality_constraint']), 0.99), 4),
                round(assess_capacity_credit(load_ts, array_sum, sum(self.parameters_yaml['cardinality_constraint']), 0.95), 4),
                round(assess_capacity_credit(load_ts, array_sum, sum(self.parameters_yaml['cardinality_constraint']), 0.90), 4))
            )

            signal_dict[c] = array_sum
            difference_dict[c] = difference
            firmness_dict[c] = firmness

        result_dict['signal'] = signal_dict
        result_dict['difference'] = difference_dict
        result_dict['firmness'] = firmness_dict
        result_dict['no_sites'] = sum(no_sites)

        return result_dict





    def optimal_locations_plot(self):

        """Plotting the optimal locations."""

        plt.clf()

        base = plot_basemap(self.outputs_pickle['coordinates_dict'])
        map = base['basemap']

        map.scatter(base['lons'], base['lats'], transform=base['projection'], marker='o', color='darkgrey', s=base['width']/1e7, zorder=2, alpha=1.0)

        tech_list = list(self.outputs_pickle['optimal_location_dict'].keys())
        tech_set = list(chain.from_iterable(combinations(tech_list, n) for n in range(1, len(tech_list)+1)))
        locations_plot = dict.fromkeys(tech_set, None)

        for key in locations_plot.keys():
            set_list = []

            for k in key:
                set_list.append(set(self.outputs_pickle['optimal_location_dict'][k]))
            locations_plot[key] = set.intersection(*set_list)

        for key in locations_plot.keys():
            proxy = set()
            init = locations_plot[key]
            subkeys = [x for x in tech_set if (x != key and len(x) > len(key))]

            if len(subkeys) > 0:

                for k in subkeys:
                    if proxy == set():
                        proxy = locations_plot[key].difference(locations_plot[k])
                    else:
                        proxy = proxy.difference(locations_plot[k])
                locations_plot[key] = list(proxy)

            else:
                locations_plot[key] = list(init)

        markers = islice(cycle(('.', 'X')), 0, len(locations_plot.keys()))
        colors = islice(cycle(('royalblue','crimson','forestgreen','goldenrod')), 0, len(locations_plot.keys()))
        for key in locations_plot.keys():

            longitudes = [i[0] for i in locations_plot[key]]
            latitudes = [i[1] for i in locations_plot[key]]
            map.scatter(longitudes, latitudes, transform=base['projection'], marker=next(markers), color=next(colors),
                        s=base['width']/(1e5), zorder=3, alpha=0.9, label=str(key))

        plt.savefig(abspath(join(self.output_folder_path,
                                 'optimal_deployment_'+str('&'.join(tuple(self.outputs_pickle['coordinates_dict'].keys())))+'.pdf')),
                                 bbox_inches='tight', dpi=300)






    def retrieve_max_locations(self):

        path_resource_data = self.parameters_yaml['path_resource_data'] + str(self.parameters_yaml['spatial_resolution']) + '/'
        horizon = self.parameters_yaml['time_slice']
        technologies = self.parameters_yaml['technologies']
        path_transfer_function_data = self.parameters_yaml['path_transfer_function_data']
        no_sites = self.parameters_yaml['cardinality_constraint']
        threshold_depth = self.parameters_yaml['depth_threshold']
        spatial_resolution = self.parameters_yaml['spatial_resolution']
        path_landseamask = self.parameters_yaml['path_landseamask']

        database = read_database(path_resource_data)
        global_coordinates = get_global_coordinates(database, self.parameters_yaml['spatial_resolution'],
                                                    self.parameters_yaml['population_density_threshold'],
                                                    self.parameters_yaml['path_population_density_data'],
                                                    self.parameters_yaml['protected_areas_selection'],
                                                    self.parameters_yaml['protected_areas_threshold'],
                                                    self.parameters_yaml['path_protected_areas_data'],
                                                    population_density_layer=False, protected_areas_layer=False)
        coordinates_filtered_depth = filter_offshore_coordinates(global_coordinates,
                                                                    threshold_depth,
                                                                    spatial_resolution,
                                                                    path_landseamask)

        suboptimal_dict = dict.fromkeys(self.parameters_yaml['regions'], None)

        location_list = []
        truncated_data_list = []
        output_data_list = []

        if len(no_sites) == 1:

            for key in suboptimal_dict.keys():

                region_coordinates = return_coordinates_from_countries(key, coordinates_filtered_depth, add_offshore=True)
                # region_coordinates = return_coordinates(key, coordinates_filtered_depth)
                truncated_data = selected_data(database, region_coordinates, horizon)

                for k in technologies:
                    tech = []
                    tech.append(k)

                    output_data = return_output(truncated_data, tech, path_transfer_function_data)[k]
                    truncated_data_sum = output_data.sum(dim='time')

                    truncated_data_list.append(truncated_data_sum)
                    output_data_list.append(output_data)

            truncated_data_concat = concat(truncated_data_list, dim='locations')
            output_data_concat = concat(output_data_list, dim='locations')

            tdata = truncated_data_concat.argsort()[-no_sites[0]:]

            for loc in tdata.values:
                location_list.append(output_data_concat.isel(locations=loc).locations.values.flatten()[0])

        else:

            raise ValueError(' Method not ready yet for partitioned problem.')

        return location_list





    def max_locations_plot(self, max_locations):

        """Plotting the optimal vs max. locations."""

        plt.clf()

        base_max = plot_basemap(self.outputs_pickle['coordinates_dict'])
        map_max = base_max['basemap']

        map_max.scatter(base_max['lons'], base_max['lats'], transform=base_max['projection'], marker='o', color='darkgrey', s=base_max['width']/1e7, zorder=2, alpha=1.0)

        longitudes = [i[0] for i in max_locations]
        latitudes = [i[1] for i in max_locations]
        map_max.scatter(longitudes, latitudes, transform=base_max['projection'], marker='.', color='royalblue',
                        s=base_max['width']/(1e5), zorder=3, alpha=0.9, label='Wind')

        plt.savefig(abspath(join(self.output_folder_path,
                                 'suboptimal_deployment_'+str('&'.join(tuple(self.outputs_pickle['coordinates_dict'].keys())))+'.pdf')),
                                 bbox_inches='tight', dpi=300)






    def optimal_locations_plot_heatmaps(self):

        """Plotting the optimal location heatmaps."""

        if (self.parameters_yaml['problem'] == 'Covering' and self.parameters_yaml['objective'] == 'budget') or \
                (self.parameters_yaml['problem'] == 'Load' and self.parameters_yaml['objective'] == 'following'):

            for it in self.outputs_pickle['deployed_capacities_dict'].keys():

                plt.clf()

                base = plot_basemap(self.outputs_pickle['coordinates_dict'])
                map = base['basemap']

                location_and_capacity_list = []
                print([val for vals in self.outputs_pickle['coordinates_dict'].values() for val in vals])
                # for idx, location in enumerate(list(set([val for vals in self.outputs_pickle['coordinates_dict'].values() for val in vals]))):
                for idx, location in enumerate([val for vals in self.outputs_pickle['coordinates_dict'].values() for val in vals]):
                    l = list(location)
                    l.append(self.outputs_pickle['deployed_capacities_dict'][it][idx])
                    location_and_capacity_list.append(l)
                print(location_and_capacity_list)
                df = pd.DataFrame(location_and_capacity_list, columns=['lon', 'lat', 'cap'])

                pl = map.scatter(df['lon'].values, df['lat'].values, transform=base['projection'], c=df['cap'].values, marker='s', s=base['width']/1e6, cmap=plt.cm.Reds, zorder=2)

                cbar = plt.colorbar(pl, ax=map, orientation= 'horizontal', pad=0.1, fraction=0.04, aspect=28)
                cbar.set_label("GW", fontsize='8')
                cbar.outline.set_visible(False)
                cbar.ax.tick_params(labelsize='x-small')

                plt.savefig(abspath(join(self.output_folder_path, 'capacity_heatmap_'+str(it)+'.pdf')), bbox_inches='tight', dpi=300)

        else:
            raise TypeError('WARNING! No such plotting capabilities for a basic deployment problem.')














