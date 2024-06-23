import numpy as np
from astropy.io import fits
from einops import rearrange
import matplotlib.pyplot as plt
import imtools as im
import copy
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_sun
from astropy.constants import R_sun
import astropy.units as u
from scipy.ndimage import rotate
from helita.io import lp


def loadFits(name, tt=0, crop=False, xrange=None, yrange=None, nan_to_num=True):
    # Load the FITS data
    datafits = fits.open(name, 'readonly')[0].data[tt, ...]

    if nan_to_num:
        # Replace NaNs with the minimum value of the non-NaN elements
        min_val = np.nanmin(datafits)
        datafits = np.nan_to_num(datafits, nan=0.999 * min_val)

    # Normalize the data to average
    qs_nom = np.nanmean(datafits[0, 0, :, :])
    if qs_nom == 0:
        raise ValueError("Normalization value (qs_nom) is zero, leading to potential division by zero.")
    datafits = rearrange(datafits, 'ns nw ny nx -> ny nx ns nw') / qs_nom

    if crop:
        if xrange is not None and yrange is not None:
            datafits = datafits[yrange[0]:yrange[1], xrange[0]:xrange[1], :, :]
        else:
            raise ValueError("Crop is set to True, but xrange or yrange is None.")

    # Check for any remaining NaNs
    if np.isnan(datafits).sum() > 0:
        raise ValueError("NaNs are present in the data after processing.")

    return np.ascontiguousarray(datafits, dtype='float64')


def get_nan_mask(name, tt=0, invert=False, crop=False, xrange=None, yrange=None):
    datafits = fits.open(name, 'readonly')[0].data[tt, ...]
    datafits = rearrange(datafits, 'ns nw ny nx -> ny nx ns nw')
    mask = np.isnan(datafits[:, :, 0, 0])  # Create a mask of nans
    if crop:
        mask = mask[yrange[0]:yrange[1], xrange[0]:xrange[1]]
    # invert the mask to select the valid pixels
    if invert:
        mask = np.invert(mask)
    return mask


def make_north_up(data, rot_fov):
    # Replace NaNs with the minimum value of the non-NaN elements
    min_val = np.nanmin(data)
    data = np.nan_to_num(data, nan=0.999 * min_val)

    # Rotate the image
    rotated_data = rotate(data, -rot_fov, reshape=True, mode='nearest')

    # Replace any NaNs introduced by rotation
    rotated_data = np.nan_to_num(rotated_data, nan=0.99 * np.nanmin(rotated_data))

    # Flip the data left to right
    rotated_data = np.fliplr(rotated_data)

    return rotated_data


def plot_crisp_image(name, tt=0, ww=0, ss=0, save_fig=False, crop=False, xtick_range=None, ytick_range=None,
                     figsize=(8, 8), fontsize=12, rot_fov=0, north_up=False, xrange=None, yrange=None):
    if ss == 0:
        label = 'I'
    elif ss == 1:
        label = 'Q'
    elif ss == 2:
        label = 'U'
    elif ss == 3:
        label = 'V'
    else:
        label = ''

    data = loadFits(name, tt, nan_to_num=True)  # ny, nx, ns, nw
    # plot data using imshow for the first wavelength and Stokes parameter
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if crop:
        final_data = data[yrange[0]:yrange[1], xrange[0]:xrange[1], ss, ww]
        if north_up:
            final_data = make_north_up(final_data, rot_fov)
        im1 = ax.imshow(final_data, cmap='Greys_r',
                        interpolation='nearest', aspect='equal', origin='lower')
    else:
        final_data = data[:, :, ss, ww]
        if north_up:
            final_data = make_north_up(final_data, rot_fov)
        im1 = ax.imshow(final_data, cmap='Greys_r', interpolation='nearest', aspect='equal', origin='lower')
    # check for nan values in the data and set it to 0.99 min value
    min_val = np.min(final_data)
    final_data = np.nan_to_num(final_data, nan=0.99*min_val)

    # get the shape of final_data
    ny, nx = final_data.shape
    cbar = fig.colorbar(im1, ax=ax, orientation='horizontal', shrink=0.8, pad=0.08)
    cbar.set_label(f'{label} ({nx}, {ny})', fontsize=1.1 * fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    if xtick_range:
        x_ticks = np.linspace(0, nx-1, num=5)
        x_labels = np.linspace(xtick_range[0], xtick_range[1], num=5)
        # ensure x_labels are displayed with 0 decimal place
        x_labels = np.round(x_labels, 0)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{label:.1f}' for label in x_labels])

    if ytick_range:
        y_ticks = np.linspace(0, ny-1, num=5)
        y_labels = np.linspace(ytick_range[0], ytick_range[1], num=5)
        # ensure y_labels are displayed with 0 decimal place
        y_labels = np.round(y_labels, 0)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f'{label:.1f}' for label in y_labels])
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    fig.tight_layout()
    if save_fig:
        print("Saving figure with data -> fig_data.pdf")
        fig.savefig('fig_data.pdf', dpi=250, format='pdf')
    plt.show()


def loadCmap(name, tt=0, crop=False, xrange=None, yrange=None):
    """
    Load the 'WCSDVARR' key from a FITS file for a specific time index and fill NaNs with 0s.

    Parameters:
    name (str): The name of the FITS file.
    tt (int): The time index.

    Returns:
    np.ndarray: The dlambda array for the specified time index.
    """
    with fits.open(name, 'readonly') as hdulist:
        # Extract the data for the specific time index tt
        dlambda = hdulist['WCSDVARR'].data[tt, 0, 0, :, :]

    # Fill NaNs with 0s
    dlambda = np.nan_to_num(dlambda)
    if crop:
        dlambda = dlambda[yrange[0]:yrange[1], xrange[0]:xrange[1]]
    return np.ascontiguousarray(dlambda, dtype='float64')


def loadFitsHeader(name):
    """
    Load the header of a FITS file.

    Parameters:
    name (str): The name of the FITS file.

    Returns:
    astropy.io.fits.header.Header: The header of the FITS file.
    """
    with fits.open(name, 'readonly') as hdulist:
        header = hdulist[0].header
    return header


def getWavelengths(name):
    io = fits.open(name)
    # Wavelength information for all wavelength points in the first time frame
    wav_output = io[1].data[0][0][0, :, 0, 0, 2] * 10  # convert from nm to Angstrom
    return np.ascontiguousarray(wav_output, dtype='float64')


def findgrid(w, dw, extra=5):
    """
    Findgrid creates a regular wavelength grid
    with a step of dw that includes all points in
    input array w. It adds extra points at the edges
    for convolution purposes

    Returns the new array and the positions of the
    wavelengths points from w in the new array
    """
    nw = np.int32(np.rint(w/dw))
    nnw = nw[-1] - nw[0] + 1 + 2*extra

    iw = np.arange(nnw, dtype='float64')*dw - extra*dw + w[0]

    idx = np.arange(w.size, dtype='int32')
    for ii in range(w.size):
        idx[ii] = np.argmin(np.abs(iw-w[ii]))

    return iw, idx


def plot_output(mos, mask, scale=0.059, save_fig=False, figsize=(30, 30)):
    """
    Plots various components of the `mos` array, applying a mask and scaling.

    Parameters:
    mos (numpy.ndarray): Multi-dimensional array with the data to plot.
    mask (numpy.ndarray): Boolean array indicating the mask.
    scale (float): Scale factor for the plot dimensions. Default is 0.059.
    save_fig (bool): Whether to save the figures as PDFs. Default is False.
    """
    # Create a deep copy of the input array to avoid modifying the original
    mos2 = copy.deepcopy(mos)

    # Update `mos2` based on `mask`
    for i in range(mos2.shape[2]):
        if i == 1:
            mos2[:, :, i][mask] = np.pi / 2
        elif i == 2:
            mos2[:, :, i][mask] = np.pi
        elif i == 3:
            mos2[:, :, i][mask] = 0
        else:
            vmax = np.percentile(mos[:, :, i][~mask], 99)
            mos2[:, :, i][mask] = 1.01 * vmax

    # Initialize the figure and axes for subplots
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=figsize)
    ax1 = ax.flatten()

    # Define colormaps and labels for the subplots
    cmaps = ['gist_gray', 'RdGy', 'gist_gray', 'bwr', 'gist_gray', 'gist_gray', 'gist_gray', 'gist_gray', 'gist_gray']
    labels = ['B [G]', 'inc [rad]', 'azi [rad]', 'Vlos [km/s]', 'vDop [Angstroms]', 'lineop', 'damp', 'S0', 'S1']

    nx, ny = mos[:, :, 0].shape
    extent = np.float32((0, nx, 0, ny)) * scale

    # Plot each component of `mos2`
    for ii in range(9):
        if ii in [1, 2]:
            a = ax1[ii].imshow(im.histo_opt(mos2[:, :, ii]), cmap=cmaps[ii], vmin=0, vmax=np.pi,
                               interpolation='nearest', extent=extent, aspect='equal', origin='lower')
        elif ii == 3:
            a = ax1[ii].imshow(im.histo_opt(mos2[:, :, ii]), cmap=cmaps[ii], vmin=-4, vmax=4,
                               interpolation='nearest', extent=extent, aspect='equal', origin='lower')
        else:
            a = ax1[ii].imshow(im.histo_opt(mos2[:, :, ii]), cmap=cmaps[ii], interpolation='nearest',
                               extent=extent, aspect='equal', origin='lower')
        cbar = fig.colorbar(a, ax=ax1[ii], orientation='horizontal', shrink=0.8, pad=0.05)

        cbar.set_label(labels[ii], fontsize=18)
        cbar.ax.tick_params(labelsize=16)

    # Remove axis labels for clarity
    for jj in range(3):
        for ii in range(3):
            if jj != 2:
                ax[jj, ii].set_xticklabels([])
            if ii != 0:
                ax[jj, ii].set_yticklabels([])
            ax[jj, ii].tick_params(axis='both', which='major', labelsize=16)  # Increase tick label size
            for label in (ax[jj, ii].get_xticklabels() + ax[jj, ii].get_yticklabels()):
                label.set_fontsize(16)  # Adjust the font size for tick labels

    fig.tight_layout()

    if save_fig:
        print("Saving figure with results -> fig_results.pdf")
        fig.savefig('fig_results.pdf', dpi=250, format='pdf')
    plt.show()


def plot_mag(mos, mask, scale=0.059, save_fig=False, vmin=None, vmax=None, figsize=(20, 10)):
    mos2 = copy.deepcopy(mos)
    # Create a new figure for Blos and Bhor maps
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Blos map
    Blos = mos2[:, :, 0] * np.cos(mos2[:, :, 1])
    Blos[mask] = 1.01 * np.percentile(Blos[~mask], 99)
    if vmin is None:
        vmin = np.percentile(Blos, 1)
    if vmax is None:
        vmax = np.percentile(Blos, 99)
    im1 = ax2[0].imshow(Blos, cmap='Greys_r', interpolation='nearest',
                        aspect='equal', vmin=vmin, vmax=vmax, origin='lower')
    ax2[0].tick_params(axis='both', which='major', labelsize=14)
    cbar1 = fig2.colorbar(im1, ax=ax2[0], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar1.set_label('Blos [G]', fontsize=18)
    cbar1.ax.tick_params(labelsize=14)

    # Bhor map
    Bhor = mos2[:, :, 0] * np.sin(mos2[:, :, 1])
    Bhor[mask] = 1.01 * np.percentile(Bhor[~mask], 99)
    if vmax is None:
        vmax = np.percentile(Bhor, 99)
    im2 = ax2[1].imshow(Bhor, cmap='Greys_r', interpolation='nearest',
                        aspect='equal', origin='lower')
    ax2[1].tick_params(axis='both', which='major', labelsize=14)
    cbar2 = fig2.colorbar(im2, ax=ax2[1], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar2.set_label('Bhor [G]', fontsize=18)
    cbar2.ax.tick_params(labelsize=14)

    fig2.tight_layout()

    if save_fig:
        print("Saving figure with results -> fig_results2.pdf")
        fig2.savefig('fig_results2.pdf', dpi=250, format='pdf')
    plt.show()


def get_fits_info(filename, verbose=False, pprint=True):
    """
    Process a FITS file to extract and compute relevant solar data.

    Parameters:
    filename (str): Path to the FITS file.

    Returns:
    tuple: A tuple containing extracted and calculated values.
    """
    # Function to calculate the distance from Earth to Sun in AU
    def get_earth_sun_distance(date):
        t = Time(date)
        sun = get_sun(t)
        return sun.distance.to(u.au).value

    # Open the FITS file
    hdr = fits.getheader(filename)
    nx = hdr['NAXIS1']
    ny = hdr['NAXIS2']
    nw = hdr['NAXIS3']
    ns = hdr['NAXIS4']
    nt = hdr['NAXIS5']
    start_time_obs = hdr['DATE-BEG']  # '2020-08-07T08:22:14'
    end_time_obs = hdr['DATE-END']
    avg_time_obs = hdr['DATE-AVG']

    # Read WCS data
    wcs_data = fits.getdata(filename, extname='WCS-TAB')

    # Extract the relevant data
    coords = np.array(wcs_data['HPLN+HPLT+WAVE+TIME'])[0]

    # Extract the coordinates
    hpln = coords[..., 0]  # (nt, nw, 2, 2) # four corners of the FOV?
    hplt = coords[..., 1]
    wave = coords[..., 2]
    time = coords[..., 3]

    # Mean pointing center of the FOV
    xcent0 = np.mean(hpln)
    ycent0 = np.mean(hplt)

    # Min max range for the Full FOV
    ln_min = np.min(hpln)
    ln_max = np.max(hpln)
    lt_min = np.min(hplt)
    lt_max = np.max(hplt)

    # Lon Lat as a function of time
    hplnt = np.unique(np.unique(hpln, axis=2), axis=1).reshape(-1, 2)
    hpltt = np.unique(np.unique(np.unique(hplt, axis=2), axis=1).reshape(-1, 2), axis=1).reshape(-1, 2)

    # Calculate R_Sun in arcseconds
    date = start_time_obs.split('T')[0]
    distance_sun_earth = get_earth_sun_distance(date)  # AU
    R_Sun_radians = np.arctan2(R_sun.to(u.au).value, distance_sun_earth)  # radians
    R_Sun_arcsec = (R_Sun_radians * u.rad).to(u.arcsec).value  # convert radians to arcseconds

    # Calculate rho and mu
    rho = np.sqrt(xcent0**2 + ycent0**2)
    mu = np.sqrt(1 - (rho / R_Sun_arcsec)**2)

    # Average wave and time on the last two dimensions
    wave2 = np.mean(wave, axis=(2, 3)) * 10  # Convert to Angstroms (nt, nw)
    time2 = np.mean(time, axis=(2, 3))

    # Convert DATE-BEG to a Time object
    start_time_astropy = Time(start_time_obs, format='isot', scale='utc')

    # Convert the `time2` values (seconds from midnight) directly to Time objects
    start_of_day = Time(f"{start_time_astropy.datetime.date()}T00:00:00", format='isot', scale='utc')
    time2_absolute = start_of_day + TimeDelta(time2, format='sec')
    time2_iso = time2_absolute.iso
    start_times = time2_iso[:, 0]
    all_start_times = [t.split('.')[0] for t in start_times]

    # # Extract specific times and central wavelength
    # start_time_calculated = time2_iso[0, 0]
    # end_time_calculated = time2_iso[-1, -1]
    # center_time_calculated = time2_iso[nt // 2, ns // 2]
    center_wavelength = wave2[0, nw // 2]
    all_wavelengths = wave2[0]

    # Convert the times to the desired format without digits after seconds
    start_time_obs_str = start_time_obs.split('.')[0]
    avg_time_obs_str = avg_time_obs.split('.')[0]
    end_time_obs_str = end_time_obs.split('.')[0]

    if pprint:
        # Print dimensions and center coordinates
        print('Dimensions:')
        print(f'  nx = {nx}')
        print(f'  ny = {ny}')
        print(f'  nw = {nw}')
        print(f'  ns = {ns}')
        print(f'  nt = {nt}')

        print('\nField of View Center Coordinates:')
        print(f'  x0 = {xcent0:.2f} (arcsec)')
        print(f'  y0 = {ycent0:.2f} (arcsec)')

        # Print the min and max range for the FOV
        print('\nField of View Range:')
        print(f'  ln_min = {ln_min:.2f} (arcsec)')
        print(f'  ln_max = {ln_max:.2f} (arcsec)')
        print(f'  lt_min = {lt_min:.2f} (arcsec)')
        print(f'  lt_max = {lt_max:.2f} (arcsec)')
        if verbose:
            print(f'  All hpln: {hplnt}')
            print(f'  All hplt: {hpltt}')

        # Print the start, end, and average times
        print('\nTime Information:')
        print(f'  Start Time     : {start_time_obs_str}')
        print(f'  Average Time   : {avg_time_obs_str}')
        print(f'  End Time       : {end_time_obs_str}')
        if verbose:
            print(f'  All Start Times: {all_start_times}')

        # Print the center wavelength and all wavelengths
        print('\nWavelength Information:')
        print(f'  Center Wavelength (Angstroms): {center_wavelength:.2f}')
        if verbose:
            print(f'  All Wavelengths   (Angstroms): {np.round(all_wavelengths, 4)}')

        # Print additional calculated values
        print('\nCalculated Values:')
        print(f'  R_Sun (arcsec) = {R_Sun_arcsec:.2f}')
        print(f'  rho            = {rho:.2f}')
        print(f'  mu             = {mu:.2f}')

    # Return all variables as a tuple
    out_dict = {
        "nx": nx,
        "ny": ny,
        "nw": nw,
        "ns": ns,
        "nt": nt,
        "xcent0": xcent0,
        "ycent0": ycent0,
        "ln_min": ln_min,
        "ln_max": ln_max,
        "lt_min": lt_min,
        "lt_max": lt_max,
        "hplnt": hplnt,
        "hpltt": hpltt,
        "start_time_obs": start_time_obs,
        "end_time_obs": end_time_obs,
        "avg_time_obs": avg_time_obs,
        "all_start_times": all_start_times,
        "center_wavelength": center_wavelength,
        "all_wavelengths": all_wavelengths,
        "R_Sun_arcsec": R_Sun_arcsec,
        "rho": rho,
        "mu": mu
    }

    return out_dict


def plot_sst_blos_bhor(blos_file, bhor_file, tt=0, xrange=None, yrange=None, figsize=(20, 10),
                       cmap='Greys_r', interpolation='nearest', aspect='equal',
                       origin='lower', colorbar_orientation='horizontal',
                       colorbar_shrink=0.8, colorbar_pad=0.05, fontsize=14,
                       blos_label='Blos [G]', bhor_label='Bhor [G]'):
    """
    Plot magnetic maps for Blos and Bhor.

    Parameters:
    - blos_file (str): Path to the Blos file.
    - bhor_file (str): Path to the Bhor file.
    - tt (int): Time index to plot.
    - xrange (list): Range of x coordinates to crop.
    - yrange (list): Range of y coordinates to crop.
    - figsize (tuple): Size of the figure.
    - cmap (str): Colormap to use for the images.
    - interpolation (str): Interpolation method for imshow.
    - aspect (str): Aspect ratio for imshow.
    - origin (str): Origin for imshow.
    - colorbar_orientation (str): Orientation of the colorbars.
    - colorbar_shrink (float): Shrink factor for the colorbars.
    - colorbar_pad (float): Padding for the colorbars.
    - fontsize (int): Font size for the labels and ticks.
    - blos_label (str): Label for the Blos colorbar.
    - bhor_label (str): Label for the Bhor colorbar.

    # Example usage
    blos_file = 'path/to/blos_file'
    bhor_file = 'path/to/bhor_file'
    xrange = [0, 100]  # example range
    yrange = [0, 100]  # example range

    plot_sst_blos_bhor(blos_file, bhor_file, xrange, yrange)
    """
    blos_sst = lp.getdata(blos_file)
    bhor_sst = lp.getdata(bhor_file)

    if xrange is not None and yrange is not None:
        blos_sst_crop = blos_sst[xrange[0]:xrange[1], yrange[0]:yrange[1], tt].T
        bhor_sst_crop = bhor_sst[xrange[0]:xrange[1], yrange[0]:yrange[1], tt].T
    else:
        blos_sst_crop = blos_sst[:, :, tt].T
        bhor_sst_crop = bhor_sst[:, :, tt].T

    # Create a new figure for Blos and Bhor maps
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    vmin = np.percentile(blos_sst_crop, 1)
    vmax = np.percentile(blos_sst_crop, 99)
    im1 = ax2[0].imshow(blos_sst_crop, cmap=cmap, interpolation=interpolation,
                        aspect=aspect, vmin=vmin, vmax=vmax, origin=origin)
    ax2[0].tick_params(axis='both', which='major', labelsize=fontsize)
    cbar1 = fig2.colorbar(im1, ax=ax2[0], orientation=colorbar_orientation,
                          shrink=colorbar_shrink, pad=colorbar_pad)
    cbar1.set_label(blos_label, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=0.8 * fontsize)

    # Bhor map
    im2 = ax2[1].imshow(bhor_sst_crop, cmap=cmap, interpolation=interpolation,
                        aspect=aspect, origin=origin)
    ax2[1].tick_params(axis='both', which='major', labelsize=fontsize)
    cbar2 = fig2.colorbar(im2, ax=ax2[1], orientation=colorbar_orientation,
                          shrink=colorbar_shrink, pad=colorbar_pad)
    cbar2.set_label(bhor_label, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=0.8 * fontsize)

    fig2.tight_layout()

    plt.show()
