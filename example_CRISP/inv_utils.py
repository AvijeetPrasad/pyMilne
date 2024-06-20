import numpy as np
from astropy.io import fits
from einops import rearrange
import matplotlib.pyplot as plt
import imtools as im
import copy


def loadFits(name, tt=0):
    datafits = fits.open(name, 'readonly')[0].data[tt, ...]
    # Fill nans with 0s:
    datafits = np.nan_to_num(datafits)
    # Normalize the data to average:
    qs_nom = np.nanmean(datafits[0, 0, :, :])
    datafits = rearrange(datafits, 'ns nw ny nx -> ny nx ns nw')/qs_nom
    return np.ascontiguousarray(datafits, dtype='float64')


def get_nan_mask(name, tt=0, invert=False):
    datafits = fits.open(name, 'readonly')[0].data[tt, ...]
    datafits = rearrange(datafits, 'ns nw ny nx -> ny nx ns nw')
    mask = np.isnan(datafits[:, :, 0, 0])  # Create a mask of nans
    # invert the mask to select the valid pixels
    if invert:
        mask = np.invert(mask)
    return mask


def loadCmap(name, tt=0):
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

    return np.ascontiguousarray(dlambda, dtype='float64')


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


def plot_output(mos, mask, scale=0.059, save_fig=False):
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
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(30, 30))
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
                               interpolation='nearest', extent=extent, aspect='equal')
        elif ii == 3:
            a = ax1[ii].imshow(im.histo_opt(mos2[:, :, ii]), cmap=cmaps[ii], vmin=-4, vmax=4,
                               interpolation='nearest', extent=extent, aspect='equal')
        else:
            a = ax1[ii].imshow(im.histo_opt(mos2[:, :, ii]), cmap=cmaps[ii], interpolation='nearest',
                               extent=extent, aspect='equal')
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

    # Create a new figure for Blos and Bhor maps
    fig2, ax2 = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    # Blos map
    Blos = mos2[:, :, 0] * np.cos(mos2[:, :, 1])
    Blos[mask] = 1.01 * np.percentile(Blos[~mask], 99)
    im1 = ax2[0].imshow(np.rot90(Blos.T), cmap='Greys_r', interpolation='nearest',
                        aspect='equal', vmin=-1500, vmax=1500)
    ax2[0].tick_params(axis='both', which='major', labelsize=14)
    cbar1 = fig2.colorbar(im1, ax=ax2[0], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar1.set_label('Blos [G]', fontsize=18)
    cbar1.ax.tick_params(labelsize=14)

    # Bhor map
    Bhor = mos2[:, :, 0] * np.sin(mos2[:, :, 1])
    Bhor[mask] = 1.01 * np.percentile(Bhor[~mask], 99)
    im2 = ax2[1].imshow(np.rot90(Bhor.T), cmap='Greys_r', interpolation='nearest',
                        aspect='equal', vmin=0, vmax=1500)
    ax2[1].tick_params(axis='both', which='major', labelsize=14)
    cbar2 = fig2.colorbar(im2, ax=ax2[1], orientation='horizontal', shrink=0.8, pad=0.05)
    cbar2.set_label('Bhor [G]', fontsize=18)
    cbar2.ax.tick_params(labelsize=14)

    fig2.tight_layout()

    if save_fig:
        print("Saving figure with results -> fig_results2.pdf")
        fig2.savefig('fig_results2.pdf', dpi=250, format='pdf')
    plt.show()
