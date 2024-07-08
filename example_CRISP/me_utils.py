import inv_utils as iu
import numpy as np
import crisp
import MilneEddington as ME
import time


def load_crisp_frame(crisp_im, tt=0, crop=False, xrange=None, yrange=None):
    """
    Load CRISP data and return a container object with wavelength array, data, and colormap.

    Parameters:
    crisp_im : str
        Path to the CRISP image file.
    tt : int
        Time step or frame index to load.
    crop : bool
        Whether to crop the image or not.
    xrange : tuple
        The range of x-coordinates to load.
    yrange : tuple
        The range of y-coordinates to load.

    Returns:
    ll : iu.container
        A container object with loaded data, wavelength array, and colormap.
    """
    ll = iu.container()
    ll.iwav = iu.get_wavelengths(crisp_im)
    ll.d = iu.load_crisp_fits(crisp_im, tt=tt, crop=crop, xrange=xrange, yrange=yrange)
    ll.cmap = iu.load_crisp_cmap(crisp_im, tt=tt, crop=crop, xrange=xrange, yrange=yrange)
    ll.mask = iu.get_nan_mask(crisp_im, tt=tt, crop=crop, xrange=xrange, yrange=yrange)
    return ll


def init_me_model(ll, sigma_strength=0.005, sigma_list=[1, 9.0, 9.0, 4.0], erh=-0.001, dtype=np.float64, nthreads=1):
    """
    Initialize the Milne-Eddington model using provided inputs.

    Parameters:
    ll : iu.container
        Container object with loaded data, including wavelength array and data cube.
    sigma_strength : array
        Array of noise estimates for the observed points.
    sigma_list : list
        List of sigma factors for Stokes Q, U, and V.
    erh : float
        Error factor for the instrumental profile.
    dtype : data type, optional
        Data type to use for calculations. Default is np.float64.
    nthreads : int, optional
        Number of threads to use. Default is 1.

    Returns:
    obs: np.ndarray
        Data cube with the fine grid dimensions.
    sig : np.ndarray
        Sigma array with the estimate of the noise for each Stokes parameter.
    l0 : float
        Central wavelength of the line.
    me : ME.MilneEddington
        Initialized Milne-Eddington model.
    """
    # Minimum step
    dw = np.min(np.diff(ll.iwav))
    dw = round(dw * 1000.) / 1000.  # avoid floating point errors

    # The inversions need to account for the instrumental
    # profile, which involve convolutions. The convolutions
    # must be done in a wavelength grid that is at least
    # 1/2 of the FWHM of the instrumental profile. In the
    # case of CRISP that would be ~55 mA / 2 = ~27.5 mA
    #
    # Get finer grid for convolutions purposes
    # Since we only observed at the lines, let's create
    # two regions, one for each line
    #
    # The observed line positions are not equidistant, the
    # Fe I 6301 points only fit into a regular grid of 5 mA
    # whereas the Fe I 6302 can fit into a 15 mA grid

    # > Find the grid for the observed wavelengths
    iw, idx = iu.find_grid(ll.iwav, dw)  # Fe I 6173
    # Now we need to create a data cube with the fine grid
    # dimensions. All observed points will contribute to the
    # inversion. The non-observed ones will have zero weight
    # but will be used internally to properly perform the
    # convolution of the synthetic spectra

    # > Create data cube with the fine grid dimensions
    ny, nx = ll.d.shape[0:2]
    obs = np.zeros((ny, nx, 4, iw.size), dtype=dtype, order='c')
    for ss in range(4):
        for ii in range(idx.size):
            obs[:, :, ss, idx[ii]] = ll.d[:, :, ss, ii]

    # > Create sigma array with the estimate of the noise for each Stokes parameter
    # Create sigma array with the estimate of the noise for
    # each Stokes parameter at all wavelengths. The extra
    # non-observed points will have a very large noise (1.e34)
    # (zero weight) compared to the observed ones (3.e-3)
    # Since the amplitudes of Stokes Q,U and V are very small
    # they have a low imprint in Chi2. We can artificially
    # give them more weight by lowering the noise estimate.
    sig = np.zeros((4, iw.size), dtype=dtype) + 1.e32
    sig[:, idx] = sigma_strength
    sig[0, idx] /= sigma_list[0]
    sig[1, idx] /= sigma_list[1]
    sig[2, idx] /= sigma_list[2]
    sig[3, idx] /= sigma_list[3]

    # > Initialize Me class
    # Init Me class. We need to create two regions with the
    # wavelength arrays defined above and a instrumental profile
    # for each region in with the same wavelength step
    tw = (np.arange(iw.size, dtype=dtype) - iw.size // 2) * dw
    # Central wavelength of the line:
    l0 = iw[iw.size // 2]
    tr = crisp.crisp(l0).dual_fpi(tw, erh=erh)

    regions = [[iw, tr / tr.sum()]]
    lines = [int(l0)]
    me = ME.MilneEddington(regions, lines, nthreads=nthreads, precision=dtype)

    return obs, sig, l0, me


def init_model(me, ny, nx,
               init_model_params=[1500, 2.2, 1.0, -0.5, 0.035, 50.0, 0.1, 0.24, 0.7], dtype=np.float64):
    """
    Initialize model parameters and repeat the model for the given dimensions.

    Parameters:
    init_model_params : list or array
        Initial model parameters [B_tot, theta_B, chi_B, gamma_B, v_los, eta_0, Doppler width, damping, s0, s1].
    dtype : data type
        Data type to use for the model parameters (np.float32 or np.float64).
    me : ME.MilneEddington
        Initialized Milne-Eddington model.
    ny : int
        Number of y-coordinates.
    nx : int
        Number of x-coordinates.

    Returns:
    Imodel : np.ndarray
        Repeated model parameters array.
    """
    if dtype == np.float32:
        iPar = np.float32(init_model_params)
    else:
        iPar = np.float64(init_model_params)

    Imodel = me.repeat_model(iPar, ny, nx)

    return Imodel


def run_randomised_me_inversion(Imodel, me, obs, sig, nRandom=6, nIter=100, chi2_thres=1, mu=1, verbose=False):
    """
    Run ME inversions of each pixel with randomizations of simple pixel-wise inversion.

    Parameters:
    Imodel : np.ndarray
        Initial model parameters array.
    me : ME.MilneEddington
        Initialized Milne-Eddington model.
    obs : np.ndarray
        Observed data array.
    sig : np.ndarray
        Sigma array with the estimate of the noise for each Stokes parameter.
    nRandom : int, optional
        Number of randomizations. Default is 6.
    nIter : int, optional
        Number of iterations. Default is 100.
    chi2_thres : float, optional
        Chi-squared threshold. Default is 1.
    mu : float, optional
        Mu value for the inversion. Default is 1.
    verbose : bool, optional
        Whether to print detailed output. Default is False.

    Returns:
    Imodel : np.ndarray
        Updated model parameters array after inversion.
    syn : np.ndarray
        Synthetic spectra.
    chi2 : np.ndarray
        Chi-squared values after inversion.
    """
    t0 = time.time()
    Imodel, syn, chi2 = me.invert(Imodel, obs, sig, nRandom=nRandom, nIter=nIter, chi2_thres=chi2_thres, mu=mu)
    t1 = time.time()
    if verbose:
        print(f"dT = {t1-t0:.2f}s -> <Chi2> = {chi2.mean():.2f}")
    return Imodel, syn, chi2


def apply_median_filter_based_on_chi2(Imodel, masked_chi2_mean,
                                      median_filter_chi2_mean_thres=[20, 50],
                                      median_filter_size=[11, 21, 31]):
    """
    Apply a median filter to the inversion model based on the masked chi-squared mean.

    Parameters:
    Imodel : np.ndarray
        Updated model parameters array after inversion.
    masked_chi2_mean : float
        Mean of chi-squared values in the masked regions.
    median_filter_chi2_mean_thres : list or array
        Threshold values for masked chi-squared mean to determine the filter size.
    median_filter_size : list or array
        Corresponding sizes for the median filter.

    Returns:
    Imodel : np.ndarray
        Model parameters array after applying the median filter.
    """
    if masked_chi2_mean < median_filter_chi2_mean_thres[0]:
        size_filter = median_filter_size[0]
    elif masked_chi2_mean < median_filter_chi2_mean_thres[1]:
        size_filter = median_filter_size[1]
    else:
        size_filter = median_filter_size[2]

    Imodel = iu.parallel_median_filter(Imodel, size_filter=size_filter)
    return Imodel


def run_spatially_regularized_inversion(me, Imodel, obs, sig,
                                        nIter=200, chi2_thres=1, mu=1,
                                        alpha_strength=30.0,
                                        alpha_list=[2, 0.5, 2, 0.01, 0.1, 0.01, 0.1, 0.01, 0.01],
                                        method=1, delay_bracket=3, dtype=np.float64, verbose=False):
    """
    Run spatially regularized inversion with the specified parameters.

    Parameters:
    Imodel : np.ndarray
        Initial model parameters array.
    me : ME.MilneEddington
        Initialized Milne-Eddington model.
    obs : np.ndarray
        Observed data array.
    sig : np.ndarray
        Sigma array with the estimate of the noise for each Stokes parameter.
    alpha_list : list or array
        List of alpha values for regularization.
    alpha_strength : float
        Strength of the alpha regularization.
    dtype : data type
        Data type to use for the alpha values (np.float32 or np.float64).
    nIter3 : int
        Number of iterations for the spatially regularized inversion.
    chi2_thres3 : float
        Chi-squared threshold for the spatially regularized inversion.
    mu : float
        Mu value for the inversion.
    method : int, optional
        Method to use for the inversion. Default is 1.
    delay_bracket : int, optional
        Delay bracket value for the inversion. Default is 3.
    verbose : bool, optional
        Whether to print detailed output. Default is False.

    Returns:
    mo : np.ndarray
        Updated model parameters array after inversion.
    syn : np.ndarray
        Synthetic spectra.
    chi2 : np.ndarray
        Chi-squared values after inversion.
    """
    t0 = time.time()

    if dtype == np.float32:
        alphas = np.float32(alpha_list)
    else:
        alphas = np.float64(alpha_list)

    mo, syn, chi2 = me.invert_spatially_regularized(
        Imodel, obs, sig, nIter=nIter, chi2_thres=chi2_thres, mu=mu,
        alpha=alpha_strength, alphas=alphas, method=method, delay_bracket=delay_bracket)

    t1 = time.time()

    if verbose:
        print(f"dT = {t1-t0:.2f}s -> <Chi2> (including regularization) = {chi2:.2f}")

    return mo, syn, chi2


def correct_velocities_for_cavity_error(mo, cmap, l0, global_offset=0.0):
    """
    Correct velocities for cavity error map from CRISP.

    Parameters:
    mo : np.ndarray
        Model parameters array with shape (1, ny, nx, 9).
    cmap : np.ndarray
        Cavity error map from CRISP.
    l0 : float
        Central wavelength of the line.
    global_offset : float, optional
        Global offset to apply to the velocity correction. Default is 0.0.

    Returns:
    mos : np.ndarray
        Corrected model parameters array with shape (ny, nx, 9).
    """
    if mo.ndim == 4:
        mos = np.squeeze(mo)  # Remove the singleton dimension in the model
    else:
        mos = mo
    mos[:, :, 3] += (cmap * 10) / l0 * 2.9e5
    if global_offset != 0.0:
        mos[:, :, 3] += global_offset

    return mos


def apply_mask_to_model(model, mask, nan_mask_replacements=[0, 0, 0, 0, 0, 0, 0, 0, 0]):
    """
    Apply a mask to the model parameters array and replace masked values.

    Parameters:
    model : np.ndarray
        Corrected model parameters array with shape (ny, nx, 9).
    mask : np.ndarray
        Mask array to apply to the model parameters.
    nan_mask_replacements : list or array
        Replacement values for each parameter in the model.

    Returns:
    masked_model : np.ndarray
        Model parameters array with masked values replaced.
    """
    masked_model = np.zeros_like(model)
    for i in range(model.shape[2]):
        masked_model[:, :, i] = iu.masked_data(
            model[:, :, i], mask, replace_val=nan_mask_replacements[i], fix_inf=True, fix_nan=True)

    return masked_model
