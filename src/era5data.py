import cdsapi
import os
from os.path import join
from helpers import read_inputs

model_parameters = read_inputs('../config_model.yml')
data_path = model_parameters['data_path']

spatial_resolution = 0.25
regions = {'EU': '75/-20/30/40'}

years = ['2017', '2018']
months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

directory = join(data_path, 'input_data/resource_data/', str(spatial_resolution))
if not os.path.exists(directory):
    os.makedirs(directory)

c = cdsapi.Client()
for region, coords in regions.items():
    for year in years:
        for month in months:
            c.retrieve(
                'reanalysis-era5-single-levels',
                {'variable':['100m_u_component_of_wind','100m_v_component_of_wind',
                             '2m_temperature', 'surface_solar_radiation_downwards', 'forecast_surface_roughness'],
                 'product_type':'reanalysis',
                 'area': str(coords),
                 'grid': str(spatial_resolution)+'/'+str(spatial_resolution),
                 'year': year,
                 'month': month,
                 'day': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                         '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                         '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31'],
                 'time': ['00:00', '01:00', '02:00', '03:00', '04:00', '05:00', '06:00', '07:00', '08:00',
                          '09:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00',
                          '18:00', '19:00', '20:00', '21:00', '22:00', '23:00'],
                 'format': 'netcdf'},
                f"{directory}/{region}_{year}_{month}.nc")

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'variable': ['low_vegetation_cover', 'high_vegetation_cover', 'land_sea_mask',
                     'model_bathymetry', 'orography', 'sea_ice_cover'],
        'product_type': 'reanalysis',
        'grid': str(spatial_resolution)+'/'+str(spatial_resolution),
        'year': '2018',
        'month': '12',
        'day': '31',
        'time': '00:00',
        'format': 'netcdf'
    },
    f"{directory}/ERA5_surface_characteristics_20181231_{str(spatial_resolution)}.nc")

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': ['orography', 'slope_of_sub_gridscale_orography'],
        'grid': str(spatial_resolution)+'/'+str(spatial_resolution),
        'year': '2018',
        'month': '12',
        'day': '31',
        'time': '00:00',
        'format': 'netcdf'
    },
    f"{directory}/ERA5_orography_characteristics_20181231_{str(spatial_resolution)}.nc")
