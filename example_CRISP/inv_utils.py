import numpy as np
from astropy.io import fits
from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imtools as im
import copy
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_sun
from astropy.constants import R_sun
import astropy.units as u
from scipy.ndimage import rotate, median_filter
from helita.io import lp
from joblib import Parallel, delayed
import os
import json
import sys
import re
import yaml
from lp_scripts.get_fov_angle import fov_angle


class container:
    def __init__(self):
        pass


def load_crisp_fits(name, tt=0, crop=False, xrange=None, yrange=None, nan_to_num=True):
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


def parallel_median_filter(Imodel, size_filter=21, n_jobs=None):
    # If n_jobs is not provided, use 90% of available CPUs
    if n_jobs is None:
        n_jobs = int(os.cpu_count() * 0.9)

    def apply_median_filter(slice_index):
        if slice_index == 2:
            sin2azi = np.sin(Imodel[:, :, 2] * 2.0)
            cos2azi = np.cos(Imodel[:, :, 2] * 2.0)
            filtered_sin2azi = median_filter(sin2azi, size=(size_filter, size_filter))
            filtered_cos2azi = median_filter(cos2azi, size=(size_filter, size_filter))
            result = 0.5 * np.arctan2(filtered_sin2azi, filtered_cos2azi)
            result[result < 0] += np.pi
            return result
        else:
            return median_filter(Imodel[:, :, slice_index], size=(size_filter, size_filter))

    # Apply the filter in parallel
    results = Parallel(n_jobs=n_jobs)(delayed(apply_median_filter)(i) for i in range(Imodel.shape[2]))

    # Combine the results back into the Imodel array
    for i in range(Imodel.shape[2]):
        Imodel[:, :, i] = results[i]

    return Imodel


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

    data = load_crisp_fits(name, tt, nan_to_num=True)  # ny, nx, ns, nw
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


def load_crisp_cmap(name, tt=0, crop=False, xrange=None, yrange=None):
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


def load_fits_data(name, ext=0):
    """
    Load the data of a FITS file.

    Parameters:
    name (str): The name of the FITS file.

    Returns:
    np.ndarray: The data of the FITS file.
    """
    with fits.open(name, 'readonly') as hdulist:
        data = hdulist[ext].data
    return data


def load_fits_header(name, out_dict=True):
    """
    Load the header of a FITS file.

    Parameters:
    name (str): The name of the FITS file.

    Returns:
    astropy.io.fits.header.Header: The header of the FITS file.
    """
    with fits.open(name, 'readonly') as hdulist:
        header = hdulist[0].header
    # convert the header to a dictionary
    if out_dict:
        header = dict(header)
    return header


def save_fits(data, header, filename, inv_comment=None, overwrite=False):
    """
    Save a FITS file with the specified data and header, including an optional comment about the data.

    Parameters:
    data (np.ndarray): The data to save.
    header (astropy.io.fits.header.Header): The header to save.
    filename (str): The name of the FITS file to save.
    comment (str): An optional comment to add to the header.
    overwrite (bool): Whether to overwrite the file if it already exists.
    """
    # Add a comment to the header if provided
    if inv_comment:
        header['INVERSION_COMMENT'] = inv_comment

    # Check if filename already exists
    if os.path.exists(filename) and not overwrite:
        print(f"File {filename} already exists. Set overwrite=True to overwrite.")
        return
    else:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(filename, overwrite=True)
        print(f"File saved: {filename}")


def get_wavelengths(name):
    io = fits.open(name)
    # Wavelength information for all wavelength points in the first time frame
    wav_output = io[1].data[0][0][0, :, 0, 0, 2] * 10  # convert from nm to Angstrom
    return np.ascontiguousarray(wav_output, dtype='float64')


def find_grid(w, dw, extra=5):
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


def plot_inversion_output(mos, mask=None, scale=0.059, save_fig=False, figsize=(30, 30),
                          apply_median_filter=False, filter_size=2, filter_index=None):
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

    if mask is not None:
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
    cmaps = ['gist_gray', 'gist_gray', 'gist_gray', 'bwr', 'gist_gray',
             'gist_gray', 'gist_gray', 'gist_gray', 'gist_gray']
    labels = ['B [G]', 'inc [rad]', 'azi [rad]', 'Vlos [km/s]', 'vDop [Angstroms]', 'lineop', 'damp', 'S0', 'S1']

    nx, ny = mos[:, :, 0].shape
    extent = np.float32((0, nx, 0, ny)) * scale

    # Plot each component of `mos2`
    for ii in range(9):
        if apply_median_filter and ii in filter_index:
            mos2[:, :, ii] = median_filter(mos2[:, :, ii], size=(filter_size, filter_size))
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


def masked_mean(data, mask):
    """
    Compute the mean of the data array, excluding the masked values.

    Parameters:
    data (numpy.ndarray): The data array.
    mask (numpy.ndarray): The mask array.

    Returns:
    float: The mean of the data array, excluding the masked values.
    """
    return np.mean(data[~mask])


def masked_stats(data, mask, pprint=True):
    """
    Compute the mean, median, and standard deviation of the data array, excluding the masked values.

    Parameters:
    data (numpy.ndarray): The data array.
    mask (numpy.ndarray): The mask array.

    Returns:
    tuple: A tuple containing the mean, median, and standard deviation of the data array, excluding the masked values.
    """
    masked_data = data[~mask]
    masked_mean = np.mean(masked_data)
    masked_median = np.median(masked_data)
    masked_std = np.std(masked_data)
    masked_max = np.max(masked_data)
    masked_min = np.min(masked_data)
    if pprint:
        print(f"Max: {masked_max:.2f},")
        print(f"Min: {masked_min:.2f}")
        print(f"Mean: {masked_mean:.2f}")
        print(f"Median: {masked_median:.2f}")
        print(f"Standard Deviation: {masked_std:.2f}")
    return masked_max, masked_min, masked_mean, masked_median, masked_std


def masked_data(data, mask, replace_val=0, fix_nan=False, fix_inf=False):
    """
    Mask the data array using the mask array, replacing the masked values with a specified value.

    Parameters:
    data (numpy.ndarray): The data array.
    mask (numpy.ndarray): The mask array.
    replace_val (float): The value to replace the masked values with.

    Returns:
    numpy.ndarray: The masked data array.
    """
    data[mask] = replace_val
    # check for nans and inf in the data and also replace them with replace_val value
    nansum = np.sum(np.isnan(data))
    infs = np.sum(np.isinf(data))
    if nansum > 0:
        print(f"Nans are present in the data in {nansum} pixels")
        if fix_nan:
            data = np.nan_to_num(data, nan=replace_val, posinf=replace_val, neginf=replace_val)
            print(f"Nans have been replaced with {replace_val}")

    if infs > 0:
        print(f"Infs are present in the data in {infs} pixels")
        if fix_inf:
            data = np.nan_to_num(data, nan=replace_val, posinf=replace_val, neginf=replace_val)
            print(f"Infs have been replaced with {replace_val}")
    return data


def plot_hist(data, bins=20, save_fig=False, figsize=(8, 8), vmin=None, vmax=None, fontsize=14,
              figname='histogram.pdf', title='Histogram', color='b', alpha=0.5, clip=False, clip_range=[None, None]):
    """
    Plot the histogram of the data array.

    Parameters:
    - data: array-like, the data to plot
    - bins: int, the number of bins for the histogram
    - save_fig: bool, whether to save the figure
    - figsize: tuple, size of the figure
    - vmin: float, minimum value for the histogram range
    - vmax: float, maximum value for the histogram range
    - fontsize: int, font size for the labels and title
    - figname: str, name of the file to save the figure
    - title: str, title of the histogram
    - color: str, color of the histogram
    - alpha: float, alpha value for the histogram fill
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    flat_data = data.flatten() if data.ndim > 1 else data
    ax.hist(flat_data, bins=bins, range=(vmin, vmax), histtype='stepfilled', color=color, alpha=alpha)

    ax.set_xlabel('Value', fontsize=fontsize)
    ax.set_ylabel('Frequency', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=0.8 * fontsize)

    fig.tight_layout()
    if save_fig:
        print(f"Saving figure with results -> {figname}")
        fig.savefig(figname, dpi=250, format='pdf')
    plt.show()


def plot_image(data, scale=1, save_fig=False, figsize=(8, 8), vmin=None, vmax=None, cutoff=0.001,
               fontsize=14, figname='image.pdf', cmap='Greys_r', title='Image', clip=False,
               xrange=None, yrange=None, show_roi=False, grid=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ny, nx = data.shape  # Note: the shape is (ny, nx)
    extent = np.float32((0, nx, 0, ny)) * scale
    if clip:
        data = np.clip(data, a_min=vmin, a_max=vmax)
    img = ax.imshow(im.histo_opt(data, cutoff=cutoff), cmap=cmap, interpolation='nearest',
                    aspect='equal', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    ax.tick_params(axis='both', which='major', labelsize=0.8 * fontsize)
    ax.set_xlabel('X [arcsec]', fontsize=fontsize)
    ax.set_ylabel('Y [arcsec]', fontsize=fontsize)
    cbar = fig.colorbar(img, ax=ax, orientation='horizontal', shrink=0.8, pad=0.10)
    cbar.set_label(title, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=0.8 * fontsize)

    # Check if xrange and yrange are provided
    if show_roi and xrange is not None and yrange is not None:
        x1, x2 = np.array(xrange) * scale
        y1, y2 = np.array(yrange) * scale
        # Add a dotted rectangle to show the region of interest
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', linestyle='--', facecolor='none')
        ax.add_patch(rect)
    if grid:
        ax.grid(True)
        # gridline in black dashed line
        ax.grid(color='black', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    if save_fig:
        print(f"Saving figure with results -> {figname}")
        fig.savefig(figname, dpi=250, format='pdf')
    plt.show()


def interactive_fov_selection(crisp_im, scale=1):
    data = load_crisp_fits(crisp_im)[:, :, 0, 0]
    ny, nx = data.shape
    x1, x2 = 0, nx-1
    y1, y2 = 0, ny-1
    plot_image(data, title='Interactive FOV Selection', cmap='gray', scale=scale, figsize=(6, 6),
               show_roi=True, xrange=[x1, x2], yrange=[y1, y2], grid=True)
    while True:
        print(f"Current FOV: x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}")
        response = input("Update FOV? (y/n): ").strip().lower()
        if response == 'yes' or response == 'y':
            x1 = int(input("x1 (bottom left): "))
            x2 = int(input("x2 (top right): "))
            y1 = int(input("y1 (bottom left): "))
            y2 = int(input("y2 (top right): "))

            xrange = [x1, x2]
            yrange = [y1, y2]

            plot_image(data, title='Interactive ROI Selection', cmap='gray', scale=scale, figsize=(6, 6),
                       show_roi=True, xrange=xrange, yrange=yrange)

        else:
            xrange = [x1, x2]
            yrange = [y1, y2]
            break
    xorg, yorg = xrange[0], yrange[0]
    xsize, ysize = xrange[1]-xrange[0], yrange[1]-yrange[0]
    return xorg, yorg, xsize, ysize


def plot_mag(mos, mask=None, scale=0.058, save_fig=False, v1min=None, v1max=None, v2max=None, figsize=(20, 10)):
    mos2 = copy.deepcopy(mos)
    nx, ny = mos2[:, :, 0].shape
    extent = np.float32((0, nx, 0, ny)) * scale
    # Create a new figure for Blos and Bhor maps
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # Blos map
    Blos = mos2[:, :, 0] * np.cos(mos2[:, :, 1])
    if mask is not None:
        Blos[mask] = 1.01 * np.percentile(Blos[~mask], 99)
    if v1min is None:
        v1min = np.percentile(Blos, 1)
    if v1max is None:
        v1max = np.percentile(Blos, 99)
    im1 = ax2[0].imshow(Blos, cmap='Greys_r', interpolation='nearest',
                        aspect='equal', vmin=v1min, vmax=v1max, origin='lower', extent=extent)
    ax2[0].tick_params(axis='both', which='major', labelsize=14)
    cbar1 = fig2.colorbar(im1, ax=ax2[0], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar1.set_label('Blos [G]', fontsize=18)
    cbar1.ax.tick_params(labelsize=14)

    # Bhor map
    Bhor = mos2[:, :, 0] * np.sin(mos2[:, :, 1])
    if mask is not None:
        Bhor[mask] = 1.01 * np.percentile(Bhor[~mask], 99)
    if v2max is None:
        v2max = np.percentile(Bhor, 95)
    im2 = ax2[1].imshow(Bhor, cmap='Greys_r', interpolation='nearest',
                        aspect='equal', origin='lower', extent=extent, vmax=v2max)
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
    central_wavelength = wave2[0, nw // 2]
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
        print(f'  Central Wavelength (Angstroms): {central_wavelength:.2f}')
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
        "central_wavelength": central_wavelength,
        "all_wavelengths": all_wavelengths,
        "R_Sun_arcsec": R_Sun_arcsec,
        "rho": rho,
        "mu": mu
    }

    return out_dict


def plot_sst_blos_bhor(blos_file, bhor_file, tt=0, xrange=None, yrange=None, figsize=(20, 10),
                       cmap='Greys_r', fontsize=14, crop=False, vmin1=None, vmax1=None, vmax2=None):
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

    if crop:
        blos_sst_crop = blos_sst[xrange[0]:xrange[1], yrange[0]:yrange[1], tt].T
        bhor_sst_crop = bhor_sst[xrange[0]:xrange[1], yrange[0]:yrange[1], tt].T
    else:
        blos_sst_crop = blos_sst[:, :, tt].T
        bhor_sst_crop = bhor_sst[:, :, tt].T

    # Create a new figure for Blos and Bhor maps
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    interpolation = 'nearest'
    origin = 'lower'
    aspect = 'equal'
    colorbar_orientation = 'horizontal'
    colorbar_shrink = 0.8
    colorbar_pad = 0.05
    blos_label = 'Blos [G]'
    bhor_label = 'Bhor [G]'
    if vmin1 is None:
        vmin1 = np.percentile(blos_sst_crop, 1)
    if vmax1 is None:
        vmax1 = np.percentile(blos_sst_crop, 99)
    im1 = ax2[0].imshow(blos_sst_crop, cmap=cmap, interpolation=interpolation,
                        aspect=aspect, vmin=vmin1, vmax=vmax1, origin=origin)
    ax2[0].tick_params(axis='both', which='major', labelsize=fontsize)
    cbar1 = fig2.colorbar(im1, ax=ax2[0], orientation=colorbar_orientation,
                          shrink=colorbar_shrink, pad=colorbar_pad)
    cbar1.set_label(blos_label, fontsize=fontsize)
    cbar1.ax.tick_params(labelsize=0.8 * fontsize)

    # Bhor map
    if vmax2 is None:
        vmax2 = np.percentile(bhor_sst_crop, 95)
    im2 = ax2[1].imshow(bhor_sst_crop, cmap=cmap, interpolation=interpolation,
                        aspect=aspect, origin=origin, vmin=0, vmax=vmax2)
    ax2[1].tick_params(axis='both', which='major', labelsize=fontsize)
    cbar2 = fig2.colorbar(im2, ax=ax2[1], orientation=colorbar_orientation,
                          shrink=colorbar_shrink, pad=colorbar_pad)
    cbar2.set_label(bhor_label, fontsize=fontsize)
    cbar2.ax.tick_params(labelsize=0.8 * fontsize)

    fig2.tight_layout()

    plt.show()


def load_crisp_fits_all_timesteps(name):
    # Load the FITS data
    hdul = fits.open(name, 'readonly')
    datafits = hdul[0].data  # Assuming the data is in the primary HDU
    nt, ns, nw, ny, nx = datafits.shape

    # Extract data for wavelength position ww=0 and stokes vector ss=0 for all time steps
    data_cube = datafits[:, 0, 0, :, :]

    # Create a mask for NaNs
    mask = np.isnan(data_cube)

    # Replace NaNs with the minimum value of the non-NaN elements
    min_val = np.nanmin(data_cube)
    data_cube = np.nan_to_num(data_cube, nan=0.999 * min_val)

    # Normalize the data to average
    qs_nom = np.nanmean(data_cube)
    if qs_nom == 0:
        raise ValueError("Normalization value (qs_nom) is zero, leading to potential division by zero.")
    data_cube /= qs_nom
    data_cube = np.ascontiguousarray(data_cube, dtype='float64')
    return data_cube, mask


def calculate_contrast(image, mask=None):
    """Calculate the contrast of a single image."""
    if mask is not None:
        # Apply the mask to the image
        image = image[~mask]

    # Calculate contrast
    mean_value = np.mean(image)
    std_dev = np.std(image)

    if mean_value == 0:
        raise ValueError("Mean value of the image is zero, leading to potential division by zero.")

    contrast = std_dev / mean_value
    return contrast


def best_contrast_frame(data_cube, mask=None):
    """
    Find the frame with the best contrast from an image data cube.

    Parameters:
    data_cube (numpy.ndarray): The image data cube of shape (ny, nx, nt).

    Returns:
    tuple: A tuple containing the image with the best contrast and a list of contrasts.
    """
    nt, ny, nx = data_cube.shape
    contrasts = []

    # Calculate the contrast for each frame
    for t in range(nt):
        contrast = calculate_contrast(data_cube[t], mask=mask[t])
        contrasts.append(contrast)

    # Find the index of the frame with the best contrast
    best_index = np.argmax(contrasts)
    best_frame = data_cube[best_index]

    return best_frame, best_index, contrasts


def plot_contrast(contrasts, figsize=(10, 6)):
    """Plot the contrast as a function of the time index."""
    plt.figure(figsize=figsize)
    plt.plot(contrasts, marker='o')
    # mark the frame with the best contrast with a red star
    best_index = np.argmax(contrasts)
    plt.plot(best_index, contrasts[best_index], 'r', marker='o', markersize=6)
    plt.title(f'Frame with Maximum Contrast: {best_index}')
    plt.xlabel('Time Index')
    plt.ylabel('Contrast')
    plt.grid(True)
    plt.show()


def remove_comments(json_string):
    # Remove // comments
    json_string = re.sub(r'\/\/.*', '', json_string)
    # Remove /* */ comments
    json_string = re.sub(r'\/\*[\s\S]*?\*\/', '', json_string)
    return json_string


def load_json_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"File not found: {config_file}")

    with open(config_file, 'r') as file:
        config = json.load(file)
    return config


def load_yaml_config(file_path):
    # check if the file exists otherwise raise an error
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def check_input_config(config, confirm=True, pprint=True):
    # Set default values for parameters
    defaults = {
        'time_index': 0,
        'best_frame_index': 0,
        'xorg': 0,
        'yorg': 0,
        'scale': 0.044,
        'is_north_up': True,
        'crop': False,
        'check_crop': False,
        'shape': 'circle',
        'hmi_con_series': 'hmi.Ic_45s',
        'hmi_mag_series': 'hmi.M_45s',
        'email': '',
        'plot_sst_pointings': False,
        'plot_hmi_ic_mag': False,
        'plot_crisp_image': False,
        'verbose': True
    }

    # Update config with default values if keys are missing
    for key, value in defaults.items():
        config.setdefault(key, value)

    required_keys = ['data_dir', 'crisp_im']

    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required configuration key '{key}'")
            sys.exit(1)

    data_dir = config['data_dir']
    # set sav_dir as data_dir if not provided
    config.setdefault('save_dir', data_dir)

    crisp_im = os.path.join(data_dir, config['crisp_im'])

    if not os.path.exists(crisp_im):
        print(f"Error: File not found '{crisp_im}'")
        sys.exit(1)

    # Get the time index with the best contrast
    data_cube, mask = load_crisp_fits_all_timesteps(crisp_im)
    best_frame, best_frame_index, contrasts = best_contrast_frame(data_cube, mask=mask)
    # update the best_frame_indes in config
    config.setdefault('best_frame_index', best_frame_index)
    # Load FITS header to get xsize and ysize if not provided
    fits_header = load_fits_header(crisp_im)
    config.setdefault('xsize', fits_header['NAXIS1'])
    config.setdefault('ysize', fits_header['NAXIS2'])

    crop = config['crop']
    check_crop = config['check_crop']
    if crop and check_crop:
        xorg, yorg, xsize, ysize = interactive_fov_selection(crisp_im, scale=1)
    else:
        xorg = config['xorg']
        xsize = config['xsize']
        yorg = config['yorg']
        ysize = config['ysize']
    time_index = config['time_index']
    scale = config['scale']
    is_north_up = config['is_north_up']
    shape = config['shape']
    save_dir = config['save_dir']
    verbose = config['verbose']
    hmi_con_series = config['hmi_con_series']
    hmi_mag_series = config['hmi_mag_series']
    email = config['email']
    plot_sst_pointings_flag = config['plot_sst_pointings_flag']
    plot_hmi_ic_mag_flag = config['plot_hmi_ic_mag_flag']
    plot_crisp_image_flag = config['plot_crisp_image_flag']

    xrange = [xorg, xorg + xsize]
    yrange = [yorg, yorg + ysize]

    # Print the parameters to verify
    if pprint:
        print("\nInput Configuration Parameters:")
        print("=" * 64)
        print(f"Data directory: {data_dir}")
        print(f"Save directory: {save_dir}")
        print(f"CRISP image   : {crisp_im}")
        print(f"Time index    : {time_index}")
        print(f"Best frame    : {best_frame_index}")
        print(f"Scale         : {scale}")
        print(f"Is North Up   : {is_north_up}")
        print(f"Shape         : {shape}")
        print(f"Crop          : {crop}")
        print(f"xorg          : {xorg}")
        print(f"yorg          : {yorg}")
        print(f"xsize         : {xsize}")
        print(f"ysize         : {ysize}")
        print(f"xrange        : {xrange}")
        print(f"yrange        : {yrange}")
        print(f"Email         : {email}")

    print("\n\nObservation Details:")
    print("=" * 64)
    fits_info = get_fits_info(crisp_im, pprint=True)
    t_obs = fits_info['avg_time_obs']
    fov = fov_angle(t_obs)
    print(f'FOV angle: {fov:.2f} deg')

    # Plot the contrast as a function of the time index
    plot_contrast(contrasts, figsize=(6, 3))
    plot_image(best_frame, title=f'Frame: {best_frame_index}', cmap='gray', scale=scale, figsize=(6, 6),
               show_roi=True, xrange=xrange, yrange=yrange)

    # Confirm the parameters from the user
    if confirm:
        validate_input = input("Do you want to proceed with these parameters? (y/n): ")
        if validate_input.lower() != 'y':
            print("Exiting the program.")
            sys.exit(0)

    # Return all the variables as a config dictionary
    config_dict = {
        'data_dir': data_dir, 'crisp_im': crisp_im, 'save_dir': save_dir,
        'xorg': xorg, 'xsize': xsize, 'yorg': yorg, 'ysize': ysize, 'time_index': time_index, 'scale': scale,
        'is_north_up': is_north_up, 'crop': crop, 'shape': shape, 'contrasts': contrasts, 'best_frame': best_frame,
        'best_frame_index': best_frame_index,
        'hmi_con_series': hmi_con_series, 'hmi_mag_series': hmi_mag_series, 'email': email, 'mask': mask,
        'fits_header': fits_header, 'fits_info': fits_info, 'fov_angle': fov,
        'plot_sst_pointings_flag': plot_sst_pointings_flag,
        'plot_hmi_ic_mag_flag': plot_hmi_ic_mag_flag, 'plot_crisp_image_flag': plot_crisp_image_flag,
        'xrange': xrange, 'yrange': yrange, 'verbose': verbose
    }
    return config_dict
