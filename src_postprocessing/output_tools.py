from scipy import sin, cos, pi, arccos
from numpy import arange, sqrt, floor, ceil
import pickle
import yaml
from os.path import join
from ast import literal_eval
from matplotlib.path import Path
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def _rect_inter_inner(x1,x2):
    n1=x1.shape[0]-1
    n2=x2.shape[0]-1
    X1=np.c_[x1[:-1],x1[1:]]
    X2=np.c_[x2[:-1],x2[1:]]
    S1=np.tile(X1.min(axis=1),(n2,1)).T
    S2=np.tile(X2.max(axis=1),(n1,1))
    S3=np.tile(X1.max(axis=1),(n2,1)).T
    S4=np.tile(X2.min(axis=1),(n1,1))
    return S1,S2,S3,S4

def _rectangle_intersection_(x1,y1,x2,y2):
    S1,S2,S3,S4=_rect_inter_inner(x1,x2)
    S5,S6,S7,S8=_rect_inter_inner(y1,y2)

    C1=np.less_equal(S1,S2)
    C2=np.greater_equal(S3,S4)
    C3=np.less_equal(S5,S6)
    C4=np.greater_equal(S7,S8)

    ii,jj=np.nonzero(C1 & C2 & C3 & C4)
    return ii,jj

def intersection(x1,y1,x2,y2):
    """
INTERSECTIONS Intersections of curves.
   Computes the (x,y) locations where two curves intersect.  The curves
   can be broken with NaNs or have vertical segments.
usage:
x,y=intersection(x1,y1,x2,y2)
    Example:
    a, b = 1, 2
    phi = np.linspace(3, 10, 100)
    x1 = a*phi - b*np.sin(phi)
    y1 = a - b*np.cos(phi)
    x2=phi
    y2=np.sin(phi)+2
    x,y=intersection(x1,y1,x2,y2)
    plt.plot(x1,y1,c='r')
    plt.plot(x2,y2,c='g')
    plt.plot(x,y,'*k')
    plt.show()
    """
    ii,jj=_rectangle_intersection_(x1,y1,x2,y2)
    n=len(ii)

    dxy1=np.diff(np.c_[x1,y1],axis=0)
    dxy2=np.diff(np.c_[x2,y2],axis=0)

    T=np.zeros((4,n))
    AA=np.zeros((4,4,n))
    AA[0:2,2,:]=-1
    AA[2:4,3,:]=-1
    AA[0::2,0,:]=dxy1[ii,:].T
    AA[1::2,1,:]=dxy2[jj,:].T

    BB=np.zeros((4,n))
    BB[0,:]=-x1[ii].ravel()
    BB[1,:]=-x2[jj].ravel()
    BB[2,:]=-y1[ii].ravel()
    BB[3,:]=-y2[jj].ravel()

    for i in range(n):
        try:
            T[:,i]=np.linalg.solve(AA[:,:,i],BB[:,i])
        except:
            T[:,i]=np.NaN


    in_range= (T[0,:] >=0) & (T[1,:] >=0) & (T[0,:] <=1) & (T[1,:] <=1)

    xy0=T[2:,in_range]
    xy0=xy0.T
    return xy0[:,0],xy0[:,1]


def clip_revenue(ts, el_price, ceiling):
    """Computes revenues associated with some synthetic timeseries.

    Parameters:

    ------------

    ts : TimeSeries
        Electricity generation time series.

    el_price : TimeSeries
        Electricity price time series

    ceiling : float
        Upper bound of electricity price, above which the value is clipped.

    Returns:

    ------------

    revenue : TimeSeries
        Time series of hourly-sampled revenue..

    """

    ts_clip = ts.where(ts <= np.quantile(ts, ceiling), 0.)
    revenue = (ts_clip * el_price).sum()

    return revenue


def assess_firmness(ts, threshold):
    """Function assessing time series "firmness".

    Parameters:

    ------------

    ts : TimeSeries
        Electricity generation time series.

    threshold : float
        Capacity factor value compared to which the firmness of the
        time series is assessed.

    Returns:

    ------------

    sequences : list
        List of integers representing the lengths of time windows with
        non-interrupted capacity factor values above "threshold".

    """

    # Replace all values smaller than the threshold with 0.
    mask = np.where(ts >= threshold, ts, 0)
    # Retrieve the indices of non-zeros from the time series.
    no_zeros = np.nonzero(mask != 0)[0]
    # Determine the length of the consecutive non-zero instances.
    sequences = [len(i) for i in np.split(no_zeros, np.where(np.diff(no_zeros) != 1)[0]+1)]

    return sequences


def assess_capacity_credit(ts_load, ts_gen, no_deployments, threshold):

    ts_load_array = ts_load.values
    ts_load_mask = np.where(ts_load_array >= np.quantile(ts_load_array, threshold), 1., 0.)
    ts_load_mask = pd.Series(data = ts_load_mask)
    ts_gen_mean = ts_gen / no_deployments
    proxy = ts_load_mask * ts_gen_mean
    proxy_nonzero = proxy.iloc[proxy.to_numpy().nonzero()[0]]

    return proxy_nonzero.mean()




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
    mapH = sqrt(c ** 2 - b ** 2)
    mapW = distsphere(min(lats), min(lons), min(lats), max(lons)) * earthRadius

    arcCenter = (mapH / 2) / earthRadius
    lat0 = update_latitude(min(lats), arcCenter)

    minlon = min(lons) - 1
    maxlon = max(lons) + 1
    minlat = min(lats) - 1
    maxlat = max(lats) + 1

    return lon0, lat0, minlon, maxlon, minlat, maxlat, mapH, mapW



def plot_basemap(coordinate_dict):
    """Creates the base of the plot functions.

    Parameters:

    ------------

    coordinate_dict : dict
        Dictionary containing coodinate pairs within regions of interest.

    Returns:

    ------------

    dict
        Dictionary containing various elements of the plot.

    """

    coordinate_list = list(set([val for vals in coordinate_dict.values() for val in vals]))

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
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), edgecolor='darkgrey', linewidth=0.5)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlabels_bottom = True
    gl.xlocator = mticker.FixedLocator(arange(floor(minlon), ceil(maxlon+10), 5))
    gl.ylocator = mticker.FixedLocator(arange(floor(minlat)-1, ceil(maxlat+10), 5))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlabel_style = {'size': 6, 'color': 'gray'}
    gl.ylabel_style = {'size': 6, 'color': 'gray'}

    ax.outline_patch.set_edgecolor('white')

    return {'basemap': ax,
            'projection': proj,
            'lons': longitudes,
            'lats': latitudes,
            'width': mapW}






def read_inputs_plotting(output_path):
    """Reads parameter file for plotting purposes.

    Parameters:

    ------------

    output_path : str
        Path towards output data.

    Returns:

    ------------

    data : dict
        Dictionary containing run parameters.

    """

    path_to_input = join(output_path, 'parameters.yml')

    with open(path_to_input) as infile:
        data = yaml.safe_load(infile)

    return data





def read_output(run_name):
    """Reads outputs for a given run.

    Parameters:

    ------------

    run_name : str
        The name of the run (given by the function init_folder in tools.py).

    Returns:

    ------------

    output_pickle : dict
        Dict-like structure containing various relevant data structures..

    """

    path_to_file = join('../output_data/', run_name, 'output_model.p')
    output_pickle = pickle.load(open(path_to_file, 'rb'))

    return output_pickle