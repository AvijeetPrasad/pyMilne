import numpy as np
from sunpy.map import make_fitswcs_header, header_helper, Map
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from sunpy.coordinates import get_body_heliographic_stonyhurst
import astropy.units as u
import interpolate2d
from sunpy.coordinates import sun
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from copy import deepcopy
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def sphere2img(lat, lon, latc, lonc, xcenter, ycenter, rsun, peff, debug=False):
    """Conversion between Heliographic coordinates to CCD coordinates.
    Ported from sphere2img written in IDL : Adapted from Cartography.c by Rick Bogart,
    by Xudong Sun [Eq 5&6 in https://arxiv.org/pdf/1309.2392.pdf]

    Parameters
    ----------
    lat, lon : array, array
        input heliographic coordinates (latitude and longitude)
    latc, lonc : float, float
        Heliographic longitude and latitude of the refenrence (center) pixel
    xcenter, ycenter : float, float
        Center coordinates in the image
    rsun : float
        Solar radius in pixels
    peff : float
        p-angle: the position angle between the geocentric north pole and the solar
        rotational north pole measured eastward from geocentric north.
    debug : bool, optional
        If True, prints debug information (default is False).

    Returns
    -------
    array
        Latitude and longitude in the new CCD coordinate system.
    """
    sin_asd = 0.004660
    cos_asd = 0.99998914
    last_latc = 0.0
    cos_latc = 1.0
    sin_latc = 0.0

    if (latc != last_latc):
        sin_latc = np.sin(latc)
        cos_latc = np.cos(latc)
        last_latc = latc

    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_lat_lon = cos_lat * np.cos(lon - lonc)

    cos_cang = sin_lat * sin_latc + cos_latc * cos_lat_lon
    r = rsun * cos_asd / (1.0 - cos_cang * sin_asd)
    xr = r * cos_lat * np.sin(lon - lonc)
    yr = r * (sin_lat * cos_latc - sin_latc * cos_lat_lon)

    cospa = np.cos(peff)
    sinpa = np.sin(peff)
    xi = xr * cospa - yr * sinpa
    eta = xr * sinpa + yr * cospa

    xi = xi + xcenter
    eta = eta + ycenter

    if debug:
        debug_info = (
            f'Debug Information (sphere2img):\n'
            f'---------------------------------\n'
            f'Input Parameters:\n'
            f'Longitude shape: {lon.shape}\n'
            f'Latitude shape: {lat.shape}\n'
            f'Longitude min, max, cent: {lon.min():.4f}, {lon.max():.4f}, {lon.mean():.4f} (radians)\n'
            f'Latitude min, max , cent: {lat.min():.4f}, {lat.max():.4f}, {lat.mean():.4f} (radians)\n'
            f'Delta lon: {lon.max()-lon.min():.4f}, Delta lat: {lat.max()-lat.min():.4f}\n'
            f'Disc Center Longitude: {lonc:.4f} (radians)\n'
            f'Disc Center Latitude: {latc:.4f} (radians)\n'
            f'Image Center (x, y): ({xcenter:.4f}, {ycenter:.4f})\n'
            f'Solar Radius: {rsun:.4f}\n'
            f'Position Angle (peff): {peff:.4f}\n'
            f'---------------------------------\n'
            f'Output Values:\n'
            f'xi shape: {xi.shape}, eta shape: {eta.shape}\n'
            f'xi min: {xi.min():.4f}, xi max: {xi.max():.4f}, xi center: {xi.mean():.4f}\n'
            f'eta min: {eta.min():.4f}, eta max: {eta.max():.4f}, eta center: {eta.mean():.4f}\n'
        )
        print(debug_info)

    return xi, eta

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def remap2cea(dict_header, field, deltal=None, debug=False, missing='nan'):
    """Map projection of the original input into the cylindical equal area system (CEA).

    Parameters
    ----------
    dict_header : dictionary
        Header with information of the observation. It works with a SDO header
        or it can be created from other data. It should include:

        dict_header = {
            'CRPIX1': float,
            'CRPIX2': float,
            'CROTA2': float,
            'CDELT1': float,
            'NAXIS1': float,
            'NAXIS2': float,
            'LONDTMIN': float,
            'LONDTMAX': float,
            'LATDTMIN': float,
            'LATDTMAX': float,
            'CRLN_OBS': float,
            'CRLT_OBS': float,
            'RSUN_OBS': float,
            'DATE-OBS': str
            }

        They should follow the same definition as given for SDO data:
        https://www.lmsal.com/sdodocs/doc?cmd=dcur&proj_num=SDOD0019&file_type=pdf

    field: array
        2D array with the magnetic field in cartesian coordinates
    deltal : float
        Heliographic degrees in the rotated coordinate system. SHARP CEA pixels are 0.03
    debug : bool, optional
        If True, prints debug information (default is False).
    missing: str/float, optional
        The method to handle missing values. Options are:
        - 'nan': Replace missing values with NaN (default)
        - 'interpolate': Interpolate missing values
        - 'nearest': Replace missing values with the nearest neighbor
        - 'zero': Replace missing values with
        - float: Replace missing values with the given value

    Returns
    -------
    array
        Remaping of the magnetic field to the cylindical equal area system (CEA).

    :Authors:
        Carlos Diaz (ISP/SU 2020), Gregal Vissers (ISP/SU 2020)
    :Modified by: Avijeet Prasad to adapt to SST datasets (2024)
    """
    # Transpose the input array
    field = field.T

    if field.shape[0] != int(dict_header['NAXIS1']) or field.shape[1] != int(dict_header['NAXIS2']):
        raise ValueError(
            f"Input field shape: {field.shape} and the header values "
            f"NAXIS1: {dict_header['NAXIS1']} and NAXIS2: {dict_header['NAXIS2']} do not match."
        )

    # Get the delta longitude by transforming the pixel size to heliographic coordinates
    if deltal is None:
        cdel_hpc = SkyCoord(dict_header['CDELT1']*u.arcsec, dict_header['CDELT1']*u.arcsec,
                            obstime=dict_header['DATE-OBS'], observer='earth', frame=frames.Helioprojective)
        cdel_hgs = cdel_hpc.transform_to(frames.HeliographicStonyhurst)
        deltal = cdel_hgs.lon.value

    if debug:
        print('Input Header Information:')
        print('---------------------------------')
        header_keys = ['CRPIX1', 'CRPIX2', 'CROTA2', 'CDELT1', 'NAXIS1', 'NAXIS2',
                       'LONDTMIN', 'LONDTMAX', 'LATDTMIN', 'LATDTMAX', 'CRLN_OBS', 'CRLT_OBS', 'RSUN_OBS', 'DATE-OBS']
        for key in header_keys:
            if key != 'DATE-OBS':
                print(f'{key}: {dict_header[key]:.4f}')
            else:
                print(f'{key}: {dict_header[key]}')
        print('---------------------------------\n')

    # Latitude at disk center [rad]
    latc = dict_header['CRLT_OBS']*np.pi/180.
    # B0 = np.copy(latc)
    # Longitude at disk center [rad]. The output is in Carrington coordinates. We use central meridian
    lonc = 0.0  # we are already subtracting CRLN_OBS when passing the input
    # L0 = 0.0
    rsun = dict_header['RSUN_OBS']

    # Position angle of rotation axis
    peff = -1.0*dict_header['CROTA2'] * np.pi / 180.0
    dx_arcsec = dict_header['CDELT1']
    rsun_px = rsun / dx_arcsec

    # Plate locations of the image center, in units of the image radius,
    # and measured from the corner. The SHARP CEA pixels have a linear dimension
    # in the x-direction of 0.03 heliographic degrees in the rotated coordinate system
    dl = deltal
    xcenter = dict_header['CRPIX1']-1  # FITS start indexing at 1, not 0
    ycenter = dict_header['CRPIX2']-1  # FITS start indexing at 1, not 0
    nlat_out = np.round((dict_header['LATDTMAX'] - dict_header['LATDTMIN'])/dl)
    # lon_max => LONDTMAX
    nlon_out = np.round((dict_header['LONDTMAX'] - dict_header['LONDTMIN'])/dl)
    nrebin = 1
    nlat_out = int(np.round(nlat_out/nrebin)*nrebin)
    nlon_out = int(np.round(nlon_out/nrebin)*nrebin)
    nx_out = nlon_out
    ny_out = nlat_out

    # latitude_out = np.arange(nlat_out)*dl + dict_header['LATDTMIN']
    # longitude_out = np.arange(nlon_out)*dl + dict_header['LONDTMIN']
    # lon_out = longitude_out[:, None] * np.pi / 180.0
    # lat_out = lonc + latitude_out[None, :] * np.pi / 180.0

    lon_center = (dict_header['LONDTMAX'] + dict_header['LONDTMIN']) / 2. * np.pi/180.0
    lat_center = (dict_header['LATDTMAX'] + dict_header['LATDTMIN']) / 2. * np.pi/180.0
    # print(lat_center,latitude_out,f[1].header['LATDTMIN'],f[1].header['LATDTMAX'])

    x_out = (np.arange(nx_out)-(nx_out-1)/2.)*dl
    y_out = (np.arange(ny_out)-(ny_out-1)/2.)*dl

    x_it = x_out[:, None] * np.pi/180.0
    y_it = y_out[None, :] * np.pi/180.0

    # sswidl plane2sphere equations for lat and lon; in sswidl these are fed
    # into sphere2img
    lat_it = np.arcsin(np.cos(lat_center)*y_it + np.sin(lat_center)*np.sqrt(1.0-y_it**2)*np.cos(x_it))
    lon_it = np.arcsin((np.sqrt(1.0-y_it**2)*np.sin(x_it)) / np.cos(lat_it)) + lon_center

    # Heliographic coordinate to CCD coordinate
    xi, eta = sphere2img(lat_it, lon_it, latc, lonc, xcenter, ycenter, rsun_px, peff, debug=debug)

    x = np.arange(dict_header['NAXIS1'])
    y = np.arange(dict_header['NAXIS2'])

    # Interpolation (or sampling)
    xi_eta = np.concatenate([xi.flatten()[:, None], eta.flatten()[:, None]], axis=1)

    field_int = interpolate2d.interpolate2d(x, y, field, xi_eta).reshape((nlon_out, nlat_out))

    if debug:
        print('\nDebug Information (remap2cea):')
        debug_info = (
            f'---------------------------------\n'
            f'Input Parameters:\n'
            f'field_x shape: {field.shape}\n'
            f'deltal: {deltal}\n'
            f'---------------------------------\n'
            f'Calculated Parameters:\n'
            f'latc: {latc:.4f} (radians)\n'
            f'lonc: {lonc:.4f} (radians)\n'
            f'rsun: {rsun:.4f}\n'
            f'peff: {peff:.4f} (radians)\n'
            f'dx_arcsec: {dx_arcsec:.4f}\n'
            f'rsun_px: {rsun_px:.4f}\n'
            f'xcenter: {xcenter:.4f}\n'
            f'ycenter: {ycenter:.4f}\n'
            f'nlat_out: {nlat_out}\n'
            f'nlon_out: {nlon_out}\n'
            f'lon_center: {lon_center:.4f} (radians)\n'
            f'lat_center: {lat_center:.4f} (radians)\n'
            f'lat_it shape: {lat_it.shape}\n'
            f'lon_it shape: {lon_it.shape}\n'
            f'lat_it min: {lat_it.min():.4f}, max: {lat_it.max():.4f}, cent: {lat_it.mean():.4f}\n'
            f'lon_it min: {lon_it.min():.4f}, max: {lon_it.max():.4f}, cent: {lon_it.mean():.4f}\n'
            f'---------------------------------\n'
            f'Output Values:\n'
            f'xi shape: {xi.shape}, eta shape: {eta.shape}\n'
            f'xi min: {xi.min():.4f}, xi max: {xi.max():.4f}, xi cent: {xi.mean():.4f}\n'
            f'eta min: {eta.min():.4f}, eta max: {eta.max():.4f}, eta cent: {eta.mean():.4f}\n'
            f'field_int shape: {field_int.shape}\n'
            f'filed_in min: {np.nanmin(field_int):.2f}, max: {np.nanmax(field_int):.2f}\n'
        )
        print(debug_info)

    field_out = field_int.T  # transpose the field_int array
    if isinstance(missing, float) or isinstance(missing, int):
        field_out = np.nan_to_num(field_out, nan=missing)
    elif missing == 'interpolate':
        field_out = fix_nan_by_interpolation(field_out)
    elif missing == 'nearest':
        field_out = np.nan_to_num(field_out, nan=np.nanmean(field_out))
    elif missing == 'zero':
        field_out = np.nan_to_num(field_out, nan=0.0)

    # Create the WCS header
    londtmax = dict_header['LONDTMAX']
    londtmin = dict_header['LONDTMIN']
    latdtmax = dict_header['LATDTMAX']
    latdtmin = dict_header['LATDTMIN']
    crln_obs = dict_header['CRLN_OBS']
    dateobs = dict_header['DATE-OBS']
    lon_c = (londtmax - londtmin) / 2.0 + londtmin + crln_obs
    lat_c = (latdtmax - latdtmin) / 2.0 + latdtmin  # + crlt_obs
    ref_coord = SkyCoord(lon_c*u.deg, lat_c*u.deg, obstime=dateobs, frame=frames.HeliographicCarrington)
    scale = [deltal, deltal] * u.deg/u.pixel
    wcs_header = make_fitswcs_header(field_out, ref_coord, scale=scale, projection_code='CEA')

    # Add observer information to the WCS header
    wcs_header['CROTA2'] = dict_header['CROTA2']
    wcs_header['CRLN_OBS'] = dict_header['CRLN_OBS']
    wcs_header['CRLT_OBS'] = dict_header['CRLT_OBS']
    wcs_header['RSUN_OBS'] = dict_header['RSUN_OBS']
    earth_observer = get_body_heliographic_stonyhurst("earth", dateobs)
    hgln_dict = header_helper.get_observer_meta(earth_observer)
    wcs_header['HGLN_OBS'] = hgln_dict['hgln_obs']
    wcs_header['HGLT_OBS'] = hgln_dict['hglt_obs']
    wcs_header['DSUN_OBS'] = hgln_dict['dsun_obs']

    # Create a SunPy map object
    field_map = Map(field_out, wcs_header)

    # Add metadata to the map object
    field_map.meta['peff'] = peff
    field_map.meta['lon_it'] = lon_it
    field_map.meta['lat_it'] = lat_it
    field_map.meta['latc'] = latc

    return field_map

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def vector_transformation(peff, latitude_out, longitude_out, B0, field_x_cea,
                          field_y_cea, field_z_cea, lat_in_rad=False, debug=False):
    """
    Magnetic field transformation matrix (see Allen Gary & Hagyard 1990)
    [Eq 7 in https://arxiv.org/pdf/1309.2392.pdf]
    """

    # Transpose the input arrays
    field_x_cea = field_x_cea.T
    field_y_cea = field_y_cea.T
    field_z_cea = field_z_cea.T

    if field_x_cea.shape != latitude_out.shape:
        raise ValueError(
            f"field_x_cea shape: {field_x_cea.shape} "
            f"\tlatitude_out shape: {latitude_out.shape} do not match."
        )

    nlat_out = len(latitude_out)
    nlon_out = len(longitude_out)

    PP = peff
    if lat_in_rad is False:
        BB = latitude_out[None, 0:nlat_out] * np.pi / 180.0
        LL = longitude_out[0:nlon_out, None] * np.pi / 180.0
    else:
        BB = latitude_out
        LL = longitude_out
    L0 = 0.0  # We use central meridian
    Ldif = LL - L0

    a11 = -np.sin(B0)*np.sin(PP)*np.sin(Ldif)+np.cos(PP)*np.cos(Ldif)
    a12 = np.sin(B0)*np.cos(PP)*np.sin(Ldif)+np.sin(PP)*np.cos(Ldif)
    a13 = -np.cos(B0)*np.sin(Ldif)
    a21 = -np.sin(BB)*(np.sin(B0)*np.sin(PP)*np.cos(Ldif)+np.cos(PP)
                       * np.sin(Ldif))-np.cos(BB)*np.cos(B0)*np.sin(PP)
    a22 = np.sin(BB)*(np.sin(B0)*np.cos(PP)*np.cos(Ldif)-np.sin(PP)
                      * np.sin(Ldif))+np.cos(BB)*np.cos(B0)*np.cos(PP)
    a23 = -np.cos(B0)*np.sin(BB)*np.cos(Ldif)+np.sin(B0)*np.cos(BB)
    a31 = np.cos(BB)*(np.sin(B0)*np.sin(PP)*np.cos(Ldif)+np.cos(PP)
                      * np.sin(Ldif))-np.sin(BB)*np.cos(B0)*np.sin(PP)
    a32 = -np.cos(BB)*(np.sin(B0)*np.cos(PP)*np.cos(Ldif)-np.sin(PP)
                       * np.sin(Ldif))+np.sin(BB)*np.cos(B0)*np.cos(PP)
    a33 = np.cos(BB)*np.cos(B0)*np.cos(Ldif)+np.sin(BB)*np.sin(B0)

    field_x_h = a11 * field_x_cea + a12 * field_y_cea + a13 * field_z_cea
    field_y_h = a21 * field_x_cea + a22 * field_y_cea + a23 * field_z_cea
    field_z_h = a31 * field_x_cea + a32 * field_y_cea + a33 * field_z_cea

    # field_z_h positive towards earth
    # field_y_h positive towards south (-field_y_h = Bt_cea)
    # field_x_h positive towards west

    field_y_h *= -1.0
    field_x_h_min = np.nanmin(field_x_h)
    field_x_h_max = np.nanmax(field_x_h)
    field_y_h_min = np.nanmin(field_y_h)
    field_y_h_max = np.nanmax(field_y_h)
    field_z_h_min = np.nanmin(field_z_h)
    field_z_h_max = np.nanmax(field_z_h)
    if debug:
        debug_info = (
            f'Debug Information (vector_transformation):\n'
            f'---------------------------------\n'
            f'Input Parameters:\n'
            f'peff: {peff:.4f}\n'
            f'latitude_out shape: {latitude_out.shape}\n'
            f'longitude_out shape: {longitude_out.shape}\n'
            f'B0: {B0:.4f}\n'
            f'field_x_cea shape: {field_x_cea.shape}\n'
            f'field_y_cea shape: {field_y_cea.shape}\n'
            f'field_z_cea shape: {field_z_cea.shape}\n'
            f'---------------------------------\n'
            f'Output Values:\n'
            f'field_x_h shape: {field_x_h.shape}\n'
            f'field_y_h shape: {field_y_h.shape}\n'
            f'field_z_h shape: {field_z_h.shape}\n'
            f'field_x_h min: {field_x_h_min:.2f}, max: {field_x_h_max:.2f}\n'
            f'field_y_h min: {field_y_h_min:.2f}, max: {field_y_h_max:.2f}\n'
            f'field_z_h min: {field_z_h_min:.2f}, max: {field_z_h_max:.2f}\n'
        )
        print(debug_info)

    return field_x_h, field_y_h, field_z_h

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def fix_nan_by_interpolation(data_array, methods=['nearest', 'linear']):
    """
    Fixes NaN regions in the input array by interpolation.

    Parameters
    ----------
    data_array : np.ndarray
        2D array with some NaN regions that need to be fixed.
    methods : list of str, optional
        List of interpolation methods to use. The first method is used for
        initial interpolation, and the second method is used for any remaining
        NaNs. Options are:
        - 'linear': Linear interpolation (default)
        - 'nearest': Nearest-neighbor interpolation
        - 'cubic': Cubic interpolation (only works for 1D and 2D data)
        Default is ['linear', 'nearest'].

    Returns
    -------
    np.ndarray
        Array with NaN regions fixed by interpolation.
    """
    # Ensure the methods list has at least two elements
    if len(methods) < 2:
        raise ValueError("Methods list must contain at least two interpolation methods.")

    # Get the shape of the data array
    rows, cols = data_array.shape

    # Create a mesh grid
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Mask NaN values
    nan_mask = np.isnan(data_array)
    valid_x = X[~nan_mask]
    valid_y = Y[~nan_mask]
    valid_data = data_array[~nan_mask]

    # Interpolation using the first method
    interpolated_data = griddata((valid_x, valid_y), valid_data, (X, Y), method=methods[0])

    # Fill any remaining NaNs using the second method
    remaining_nan_mask = np.isnan(interpolated_data)
    if np.any(remaining_nan_mask):
        interpolated_data[remaining_nan_mask] = griddata(
            (valid_x, valid_y), valid_data, (X[remaining_nan_mask], Y[remaining_nan_mask]), method=methods[1]
        )

    return interpolated_data


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def bvec2cea(dict_header, field_x, field_y, field_z, deltal=None, debug=False, missing='nan'):
    """Transformation to Cylindrical equal area projection (CEA) from CCD
    detector as it is done with SHARPs according to Xudong Sun (2018).

    Parameters
    ----------
    field_x, field_y, field_z: array
        2D array with the magnetic field in cartesian coordinates

    dict_header : dictionary
        Header with information of the observation. It works with a SDO header
        or it can be created from other data. It should include:

        dict_header = {
            'CRPIX1': float,
            'CRPIX2': float,
            'CROTA2': float,
            'CDELT1': float,
            'NAXIS1': float,
            'NAXIS2': float,
            'LONDTMIN': float,
            'LONDTMAX': float,
            'LATDTMIN': float,
            'LATDTMAX': float,
            'CRLN_OBS': float,
            'CRLT_OBS': float,
            'RSUN_OBS': float,
            'DATE-OBS': str
            }

        They should follow the same definition as given for SDO data:
        https://www.lmsal.com/sdodocs/doc?cmd=dcur&proj_num=SDOD0019&file_type=pdf

    deltal : float
        Heliographic degrees in the rotated coordinate system. SHARP CEA pixels are 0.03

    debug : bool, optional
        If True, prints debug information (default is False).
    missing: str/float, optional
        The method to handle missing values. Options are:
        - 'nan': Replace missing values with NaN (default)
        - 'interpolate': Interpolate missing values
        - 'nearest': Replace missing values with the nearest neighbor
        - 'zero': Replace missing values with zeros
        - float: Replace missing values with the given value

    Returns
    -------
    arrays
        Three components of the magnetic field in heliocentric spherical coordinates and
        in cylindrical equal area projection.

    Example
    -------
    >>> fz = field * np.cos(inclination * np.pi / 180.0)
    >>> fhor = field * np.sin(inclination * np.pi / 180.0)
    >>> fy = field_hor * np.cos(azimuth * np.pi / 180.0)
    >>> fx = -field_hor * np.sin(azimuth * np.pi / 180.0)
    >>> delta_l = 0.03
    >>> bx, by, bz = bvec2cea(file.header, fx, fy, fz, delta_l)

    :Authors:
        Carlos Diaz (ISP/SU 2020), Gregal Vissers (ISP/SU 2020)
    :Modified by: Avijeet Prasad to adapt to SST datasets (2024)
    """

    # Get the delta longitude by transforming the pixel size to heliographic coordinates
    if deltal is None:
        cdel_hpc = SkyCoord(dict_header['CDELT1']*u.arcsec, dict_header['CDELT1']*u.arcsec,
                            obstime=dict_header['DATE-OBS'], observer='earth', frame=frames.Helioprojective)
        cdel_hgs = cdel_hpc.transform_to(frames.HeliographicStonyhurst)
        deltal = cdel_hgs.lon.value

    # Map projection
    field_x_map = remap2cea(dict_header, field_x, deltal, debug=debug)
    field_y_map = remap2cea(dict_header, field_y, deltal, debug=debug)
    field_z_map = remap2cea(dict_header, field_z, deltal, debug=debug)

    peff = field_x_map.meta['peff']
    lon_it = field_x_map.meta['lon_it']
    lat_it = field_x_map.meta['lat_it']
    latc = field_x_map.meta['latc']

    field_x_int = field_x_map.data
    field_y_int = field_y_map.data
    field_z_int = field_z_map.data

    if debug:
        debug_info = (
            f'Debug Information (remap2cea):\n'
            f'---------------------------------\n'
            f'peff: {peff}\n'
            f'lon_it shape: {lon_it.shape}, lat_it shape: {lat_it.shape}\n'
            f'latc: {latc:.4f}\n'
            f'field_x_int shape: {field_x_int.shape}\n'
            f'field_y_int shape: {field_y_int.shape}\n'
            f'field_z_int shape: {field_z_int.shape}\n'
        )
        print(debug_info)

    # Vector transformation
    field_x_h, field_y_h, field_z_h = vector_transformation(
        peff, lat_it, lon_it, latc, field_x_int, field_y_int, field_z_int, lat_in_rad=True, debug=debug)

    # Handle missing values
    if isinstance(missing, float) or isinstance(missing, int):
        field_x_h = np.nan_to_num(field_x_h, nan=missing)
        field_y_h = np.nan_to_num(field_y_h, nan=missing)
        field_z_h = np.nan_to_num(field_z_h, nan=missing)
    elif missing == 'interpolate':
        field_x_h = fix_nan_by_interpolation(field_x_h)
        field_y_h = fix_nan_by_interpolation(field_y_h)
        field_z_h = fix_nan_by_interpolation(field_z_h)
    elif missing == 'nearest':
        field_x_h = np.nan_to_num(field_x_h, nan=np.nanmean(field_x_h))
        field_y_h = np.nan_to_num(field_y_h, nan=np.nanmean(field_y_h))
        field_z_h = np.nan_to_num(field_z_h, nan=np.nanmean(field_z_h))
    elif missing == 'zero':
        field_x_h = np.nan_to_num(field_x_h, nan=0.0)
        field_y_h = np.nan_to_num(field_y_h, nan=0.0)
        field_z_h = np.nan_to_num(field_z_h, nan=0.0)

    if debug:
        debug_info = (
            f'Debug Information (vector_transformation):\n'
            f'---------------------------------\n'
            f'field_x_h shape: {field_x_h.shape}\n'
            f'field_y_h shape: {field_y_h.shape}\n'
            f'field_z_h shape: {field_z_h.shape}\n'
        )
        print(debug_info)

    # Create the WCS header and return the magnetic field components
    londtmax = dict_header['LONDTMAX']
    londtmin = dict_header['LONDTMIN']
    latdtmax = dict_header['LATDTMAX']
    latdtmin = dict_header['LATDTMIN']
    crln_obs = dict_header['CRLN_OBS']
    # crlt_obs = dict_header['CRLT_OBS']
    dateobs = dict_header['DATE-OBS']
    lon_c = (londtmax - londtmin) / 2.0 + londtmin + crln_obs
    lat_c = (latdtmax - latdtmin) / 2.0 + latdtmin  # + crlt_obs
    ref_coord = SkyCoord(lon_c*u.deg, lat_c*u.deg, obstime=dateobs, frame=frames.HeliographicCarrington)
    scale = [deltal, deltal] * u.deg/u.pixel
    wcs_header = make_fitswcs_header(field_z_h.T, ref_coord, scale=scale, projection_code='CEA')
    wcs_header['CRLN_OBS'] = dict_header['CRLN_OBS']
    wcs_header['CRLT_OBS'] = dict_header['CRLT_OBS']
    wcs_header['RSUN_OBS'] = dict_header['RSUN_OBS']

    earth_observer = get_body_heliographic_stonyhurst("earth", dateobs)
    hgln_dict = header_helper.get_observer_meta(earth_observer)
    wcs_header['HGLN_OBS'] = hgln_dict['hgln_obs']
    wcs_header['HGLT_OBS'] = hgln_dict['hglt_obs']
    wcs_header['DSUN_OBS'] = hgln_dict['dsun_obs']

    if debug:
        # print the WCS header
        print('WCS Header:')
        print('---------------------------------')
        for key, value in wcs_header.items():
            print(f'{key}: {value}')
    # Return the magnetic field components and the WCS header
    fx = field_x_h.T   # positive towards west
    fy = -field_y_h.T  # positive towards south
    fz = field_z_h.T   # positive towards earth
    return fx, fy, fz, wcs_header


def get_wcs_info(map, verbose=False):
    """
    Extract WCS information from a SunPy map object.

    Parameters
    ----------
    map : sunpy.map.Map
        The SunPy map object from which to extract WCS information.
    verbose : bool, optional
        If True, prints the WCS dictionary. Default is False.

    Returns
    -------
    wcs_dict : dict
        Dictionary containing WCS information.

    # Example usage:
    wcs_dict = extract_wcs_info(map, verbose=True)
    """
    # Extracting the WCS information from the map metadata
    dateobs = map.meta['date-obs']
    cdelt1 = map.scale.axis1
    cdelt2 = map.scale.axis2  # cdelt2 = cdelt1, if not specified
    crota2 = map.meta['CROTA2']
    naxis1 = map.data.shape[1]
    naxis2 = map.data.shape[0]
    center = map.center

    # Get the coordinates of the bottom left and top right corners of the map
    tx1 = center.Tx + ((naxis1 - 1) / 2.0) * u.pix * cdelt1
    ty1 = center.Ty + ((naxis2 - 1) / 2.0) * u.pix * cdelt2
    tx2 = center.Tx - ((naxis1 - 1) / 2.0) * u.pix * cdelt1
    ty2 = center.Ty - ((naxis2 - 1) / 2.0) * u.pix * cdelt2

    blc_hpc = SkyCoord(tx1, ty1, obstime=dateobs, observer='earth', frame=frames.Helioprojective)
    trc_hpc = SkyCoord(tx2, ty2, obstime=dateobs, observer='earth', frame=frames.Helioprojective)

    # Convert the bottom left and top right coordinates to Heliographic Carrington
    blc_carr = blc_hpc.transform_to(frames.HeliographicCarrington)
    trc_carr = trc_hpc.transform_to(frames.HeliographicCarrington)

    # Get the center of the solar disk in Helioprojective and Heliographic Carrington coordinates
    disc_center_hpc = SkyCoord(0 * u.arcsec, 0 * u.arcsec, obstime=dateobs,
                               observer='earth', frame=frames.Helioprojective)
    disc_center_carr = disc_center_hpc.transform_to(frames.HeliographicCarrington)
    r_sun = sun.angular_radius(dateobs).value

    # Calculate the minimum and maximum longitude and latitude of the map
    lon_min = min(blc_carr.lon.value - disc_center_carr.lon.value, trc_carr.lon.value - disc_center_carr.lon.value)
    lon_max = max(blc_carr.lon.value - disc_center_carr.lon.value, trc_carr.lon.value - disc_center_carr.lon.value)
    lat_min = min(blc_carr.lat.value, trc_carr.lat.value)
    lat_max = max(blc_carr.lat.value, trc_carr.lat.value)

    # Create a dictionary with the WCS information
    wcs_dict = {}
    wcs_dict['CRPIX1'] = (blc_hpc.Tx / cdelt1 + 1 * u.pix).value
    wcs_dict['CRPIX2'] = (blc_hpc.Ty / cdelt1 + 1 * u.pix).value
    wcs_dict['CDELT1'] = cdelt1.value
    wcs_dict['CROTA2'] = crota2
    wcs_dict['NAXIS1'] = naxis1
    wcs_dict['NAXIS2'] = naxis2
    wcs_dict['CRLN_OBS'] = disc_center_carr.lon.value
    wcs_dict['CRLT_OBS'] = disc_center_carr.lat.value
    wcs_dict['LONDTMIN'] = lon_min
    wcs_dict['LONDTMAX'] = lon_max
    wcs_dict['LATDTMIN'] = lat_min
    wcs_dict['LATDTMAX'] = lat_max
    wcs_dict['RSUN_OBS'] = r_sun
    wcs_dict['DATE-OBS'] = dateobs

    # Print the WCS dictionary if verbose is True
    if verbose:
        for key, value in wcs_dict.items():
            if key != 'DATE-OBS':
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    return wcs_dict


def make_map(data, crval1, crval2, cdelt1, crota2, date_obs,
             cdelt2=None, ctype1='HPLN-TAN', ctype2='HPLT-TAN', telescope=None,
             instrument=None, wavelength=6173, verbose=False):
    """
    Generate the FITS WCS header from the given WCS information.

    Parameters
    ----------
   data: np.ndarray
        2D array with the data.
    crval1 : float
        Patch center in arcseconds (longitude).
    crval2 : float
        Patch center in arcseconds (latitude).
    cdelt1 : float
        Arcseconds per pixel.
    crota2 : float
        Rotation of solar north in degrees.
    date_obs : str
        Observation date in 'YYYY-MM-DDTHH:MM:SS' format.
    ctype1 : str, optional
        Type of coordinate along the X-axis. Default is 'HPLN-TAN'.
    ctype2 : str, optional
        Type of coordinate along the Y-axis. Default is 'HPLT-TAN'.
    telescope : str, optional
        Name of the telescope. Default is 'SDO'.
    instrument : str, optional
        Name of the instrument. Default is 'HMI'.
    wavelength : int, optional
        Observation wavelength in angstroms. Default is 6173.
    verbose : bool, optional
        If True, prints the WCS header. Default is False.

    Returns
    -------
    header : sunpy.map.header
        Generated FITS WCS header.

    # Example usage:
    data = np.random.rand(288, 472)
    crval1 = -495
    crval2 = 310
    cdelt1 = 0.504
    crota2 = 180.0
    date_obs = '2020-08-07T06:00:00'
    sunpy_map = make_map(data, crval1, crval2, cdelt1, crota2, date_obs, verbose=True)
    sunpy_map.peek()
    """
    if cdelt2 is None:
        cdelt2 = cdelt1  # cdelt2 is the same as cdelt1

    # Get the shape of the data array
    naxis2, naxis1 = data.shape

    # Define the WCS metadata
    map_meta = {
        'crval1': crval1,
        'crval2': crval2,
        'cdelt1': cdelt1,
        'cdelt2': cdelt2,
        'crota2': crota2,
        'date-obs': date_obs,
        'ctype1': ctype1,
        'ctype2': ctype2,
        'crpix1': (naxis1 + 1) / 2,
        'crpix2': (naxis2 + 1) / 2,
        'naxis1': naxis1,
        'naxis2': naxis2
    }

    projection = ctype1.split('-')[1]
    if ctype1 == 'HPLN-TAN':
        # Create the coordinate for the reference pixel
        coord = SkyCoord(map_meta['crval1'] * u.arcsec, map_meta['crval2'] * u.arcsec,
                         obstime=map_meta['date-obs'], observer='earth', frame=frames.Helioprojective)

    elif ctype1 == 'CRLN-CEA':
        coord = SkyCoord(map_meta['crval1'] * u.deg, map_meta['crval2'] * u.deg,
                         obstime=map_meta['date-obs'], observer='earth', frame=frames.HeliographicCarrington)
    # Generate the FITS WCS header
    header = make_fitswcs_header((naxis2, naxis1), coord,
                                 reference_pixel=[map_meta['crpix1'], map_meta['crpix2']] * u.pixel,
                                 scale=[map_meta['cdelt1'], map_meta['cdelt2']] * u.arcsec / u.pixel,
                                 telescope=telescope, instrument=instrument,
                                 rotation_angle=map_meta['crota2'] * u.deg,
                                 wavelength=wavelength * u.angstrom, projection_code=projection)
    # Add the crota2 keyword to the header
    header['crota2'] = crota2
    if verbose:
        print("Generated FITS WCS Header:")
        for key, value in header.items():
            print(f"{key}: {value}")
    sunpy_map = Map(data, header)
    return sunpy_map


def plot_map_on_grid(map, figsize=(8, 8), coord_limits=[-1024, 1024], limb_color='k', grid_color='black',
                     grid_alpha=0.5, grid_lw=0.5, coord_marker='o', coord_color='red', coord_size=0.01,
                     vmin_percentile=0.5, vmax_percentile=99.5, cent_coord_size=0.3, return_fig=False,
                     project_dc=False, cmap='gray'):
    """
    Plot a SunPy map on a full solar grid.

    Parameters
    ----------
    map : sunpy.map.Map
        The SunPy map object to plot.
    figsize : tuple, optional
        Size of the figure. Default is (8, 8).
    coord_limits : list, optional
        Limits for the coordinates in arcseconds. Default is [-1024, 1024].
    limb_color : str, optional
        Color of the solar limb. Default is 'k'.
    grid_color : str, optional
        Color of the grid lines. Default is 'black'.
    grid_alpha : float, optional
        Alpha value of the grid lines. Default is 0.5.
    grid_lw : float, optional
        Line width of the grid lines. Default is 0.5.
    coord_marker : str, optional
        Marker style for the coordinates. Default is 'o'.
    coord_color : str, optional
        Color of the coordinate markers. Default is 'red'.
    coord_size : float, optional
        Size of the coordinate markers. Default is 0.01.
    vmin_percentile : float, optional
        Percentile for the minimum data value to plot. Default is 0.5.
    vmax_percentile : float, optional
        Percentile for the maximum data value to plot. Default is 99.95.
    project_dc : bool, optional
        If True, project the map to disc center. Default is False.
    cmap : str, optional
        Colormap to use. Default is 'gray'.
    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    ax : matplotlib.axes._subplots.AxesSubplot
        The matplotlib axes object.
    # Example usage:
        # plot_full_solar_grid(bz_map)
    """
    date = map.date
    scale = map.scale
    data = np.full((10, 10), np.nan)
    cent_coord = SkyCoord(0*u.arcsec, 0*u.arcsec, obstime=date, observer='earth', frame=frames.Helioprojective)
    cent_crln = cent_coord.transform_to(frames.HeliographicCarrington)
    temp_header = make_fitswcs_header(
        data, cent_coord, scale=[scale.axis1.value, scale.axis2.value]*u.arcsec/u.pixel, reference_pixel=[0, 0]*u.pixel)
    blank_map = Map(data, temp_header)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection=blank_map)
    if project_dc:
        # deepcopy the map to avoid changing the original map
        map2 = deepcopy(map)
        map2.meta['CRVAL1'] = cent_crln.lon.value
        map2.meta['CRVAL2'] = cent_crln.lat.value
        map2.meta['CRPIX1'] = (map2.meta['NAXIS1'] + 1) / 2.0
        map2.meta['CRPIX2'] = (map2.meta['NAXIS2'] + 1) / 2.0
        # Create a new map with the updated metadata
        map2 = Map(map2.data, map2.meta)
    map_center = map.center.transform_to(frames.Helioprojective(observer='earth', obstime=date))
    blank_map.plot(axes=ax)
    blank_map.draw_limb(axes=ax, color=limb_color)
    blank_map.draw_grid(axes=ax, color=grid_color, alpha=grid_alpha, lw=grid_lw, system='carrington')

    xc = coord_limits * u.arcsec
    yc = coord_limits * u.arcsec
    coords = SkyCoord(xc, yc, frame=blank_map.coordinate_frame)
    ax.plot_coord(coords, coord_marker, color=coord_color, markersize=coord_size)
    ax.plot_coord(cent_coord, coord_marker, color=coord_color, markersize=cent_coord_size)
    ax.plot_coord(map_center, coord_marker, color=coord_color, markersize=cent_coord_size,
                  label=f'Map center: ({map_center.Tx.value:.0f}, {map_center.Ty.value:.0f})')
    map.plot(axes=ax, autoalign=True, zorder=1, vmin=np.nanpercentile(
        map.data, vmin_percentile), vmax=np.nanpercentile(map.data, vmax_percentile), title=date.iso[:-4], cmap=cmap)
    if project_dc:
        map2.plot(axes=ax, autoalign=True, zorder=1, vmin=np.nanpercentile(
            map2.data, vmin_percentile), vmax=np.nanpercentile(map2.data, vmax_percentile),
            title=date.iso[:-4], cmap=cmap)
    ax.legend(loc="upper left", markerscale=coord_size, markerfirst=False, frameon=False)
    plt.show()
    if return_fig:
        return fig, ax
    else:
        return None
