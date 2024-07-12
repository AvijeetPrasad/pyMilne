# %% [markdown]
# ### Milne Eddington Inversion of SST/CRISP data

# %%
# Importing the required libraries, custom libraries and setting the python path
import importlib
from load_env_and_set_pythonpath import print_pythonpath
import numpy as np
from einops import rearrange
from uncertainties import unumpy
import os
import inv_utils as iu
import me_utils as meu
import helita_io_lp as lp
import hmi_plot as hp
print_pythonpath()

# %%
# Reload the libraries to get the changes
importlib.reload(iu)
importlib.reload(meu)
print("All libraries reloaded successfully\n")

# %%
# Load the configuration from the JSON file
input_config = iu.load_yaml_config('input_config.yaml')
# Check the input configuration
config = iu.check_input_config(input_config, pprint=True, confirm=False)

# %%
# # Extract the input parameters
data_dir = config['data_dir']
save_dir = config['save_dir']
crisp_im = config['crisp_im']
xorg = config['xorg']
xsize = config['xsize']
yorg = config['yorg']
ysize = config['ysize']
xrange = config['xrange']
yrange = config['yrange']
time_range = config['time_range']
best_frame_index = config['best_frame_index']
scale = config['scale']
is_north_up = config['is_north_up']
crop = config['crop']
shape = config['shape']
best_frame = config['best_frame']
contrasts = config['contrasts']
hmi_con_series = config['hmi_con_series']
hmi_mag_series = config['hmi_mag_series']
email = config['email']
fov_angle = config['fov_angle']
plot_sst_pointings_flag = config['plot_sst_pointings_flag']
plot_hmi_ic_mag_flag = config['plot_hmi_ic_mag_flag']
plot_crisp_image_flag = config['plot_crisp_image_flag']
verbose = config['verbose']
inversion_save_fits_list = config['inversion_save_fits_list']
inversion_save_errors_fits = config['inversion_save_errors_fits']
inversion_save_lp_list = config['inversion_save_lp_list']
inversion_save_errors_lp = config['inversion_save_errors_lp']
# union of inversion_save_fits_list and inversion_save_lp_list
inversion_save_list = list(set(inversion_save_fits_list + inversion_save_lp_list))
wfa_blos_map = config['wfa_blos_map']

# Extract the fits information from the header for the best frame
tt = best_frame_index
fits_info = config['fits_info']
nx = fits_info['nx']
ny = fits_info['ny']
nw = fits_info['nw']
nt = fits_info['nt']
mu = fits_info['mu']
x1 = fits_info['hplnt'][tt][0]
x2 = fits_info['hplnt'][tt][1]
y1 = fits_info['hpltt'][tt][0]
y2 = fits_info['hpltt'][tt][1]
tobs = fits_info['all_start_times'][tt]
tstart = fits_info['start_time_obs']
tend = fits_info['end_time_obs']
hplnt = fits_info['hplnt']
hpltt = fits_info['hpltt']
all_start_times = fits_info['all_start_times']
central_wavelength = fits_info['central_wavelength']

# Reset the x and y ranges if cropping is enabled
if crop:
    x_list = np.linspace(x1, x2, num=nx)
    y_list = np.linspace(y1, y2, num=ny)
    x_list = x_list[xrange[0]:xrange[1]]
    y_list = y_list[yrange[0]:yrange[1]]
    x1 = x_list[0]
    x2 = x_list[-1]
    y1 = y_list[0]
    y2 = y_list[-1]
    nx = xsize
    ny = ysize
else:
    print('No cropping is done\n')

# Load the fits header as a dictionary
fits_header = iu.load_fits_header(crisp_im, out_dict=False)
fits_header_dict = iu.load_fits_header(crisp_im, out_dict=True)

# %%
if plot_sst_pointings_flag:
    hp.plot_sst_pointings(tstart, hmi_con_series, hplnt, hpltt, figsize=(6, 6), email=email, save_dir=save_dir)

# %%
if plot_hmi_ic_mag_flag:
    hp.plot_hmi_ic_mag(tobs, hmi_con_series, hmi_mag_series, email, x1, x2, y1, y2, save_dir=save_dir,
                       figsize=(10, 5),  is_north_up=is_north_up, fov_angle=fov_angle, shape=shape)

# %%
if plot_crisp_image_flag:
    print('SST CRISP image with North up:', not (is_north_up))
    iu.plot_crisp_image(crisp_im, tt=best_frame_index, ss=0, ww=0, figsize=(8, 8), fontsize=10, rot_fov=fov_angle,
                        north_up=not (is_north_up), crop=crop, xrange=xrange, yrange=yrange,
                        xtick_range=[x1, x2], ytick_range=[y1, y2])

    iu.plot_crisp_image(crisp_im, tt=best_frame_index, ss=3, ww=nw // 4, figsize=(8, 8), fontsize=10, rot_fov=fov_angle,
                        north_up=not (is_north_up), crop=crop, xrange=xrange, yrange=yrange,
                        xtick_range=[x1, x2], ytick_range=[y1, y2])

# %%
# Load the variables from the inversion configuration
inversion_config = iu.load_yaml_config('inversion_config.yaml')

dtype = inversion_config['dtype']
sigma_strength = inversion_config['sigma_strength']
sigma_list = inversion_config['sigma_list']
erh = inversion_config['erh']
init_model_params = inversion_config['init_model_params']
nRandom1 = inversion_config['nRandom1']
nIter1 = inversion_config['nIter1']
chi2_thres1 = inversion_config['chi2_thres1']
median_filter_chi2_mean_thres = inversion_config['median_filter_chi2_mean_thres']
median_filter_size = inversion_config['median_filter_size']
nRandom2 = inversion_config['nRandom2']
nIter2 = inversion_config['nIter2']
chi2_thres2 = inversion_config['chi2_thres2']
nIter3 = inversion_config['nIter3']
chi2_thres3 = inversion_config['chi2_thres3']
alpha_strength = inversion_config['alpha_strength']
alpha_list = inversion_config['alpha_list']
nan_mask_replacements = inversion_config['nan_mask_replacements']
nthreads = iu.get_nthreads()

# %% [markdown]
# ##### Run the inversion for the best time index to get the initial guess for the Milne Eddington inversion

# %%
# List of variables obtained after the final inversion
inversion_out_list = ["Bstr", "Binc", "Bazi", "Vlos", "Vdop", "etal", "damp", "S0", "S1", "Blos", "Bhor", "Nan_mask"]
inverstion_error_out_list = ["Bstr_err", "Binc_err", "Bazi_err", "Vlos_err", "Vdop_err",
                             "etal_err", "damp_err", "S0_err", "S1_err", "Blos_err", "Bhor_err", "Nan_mask"]

# %%
first_iteration = True
count = 1
for tt in time_range:
    nt = len(time_range)
    print(f'\n\n=== Processing Frame: {count}/{nt}, Index: {tt}, Time: {all_start_times[tt]} UT ===')
    count += 1

    # Load the CRISP image for a given time step
    ll = meu.load_crisp_frame(crisp_im, tt, crop=crop, xrange=xrange, yrange=yrange)

    # Setup the inversion parameters for the ME inversion
    obs, sig, l0, me = meu.init_me_config(ll, sigma_strength, sigma_list, erh=erh, dtype=dtype, nthreads=nthreads)

    if first_iteration:
        # Obtain the initial model parameters after the inversion
        Imodel = meu.init_model(me, ny, nx, init_model_params=init_model_params, dtype=dtype)

    # Run the randomised ME inversion for the first time
    print('=== BLOCK 1: Randomised ME Inversions ===')
    Imodel, syn, chi2 = meu.run_randomised_me_inversion(
        Imodel, me, obs, sig, nRandom=nRandom1, nIter=nIter1, chi2_thres=chi2_thres1, mu=mu, verbose=verbose)
    masked_chi2_mean = iu.masked_mean(chi2, ll.mask)
    if verbose:
        print(f'Masked chi2 mean: {masked_chi2_mean:.2f}')
        iu.plot_inversion_output(Imodel, ll.mask, scale=scale, save_fig=False)
        iu.plot_mag(Imodel, ll.mask, scale=scale, save_fig=False)

    # if not init_model_from_sequence:
    if not first_iteration:
        median_filter_size = [2, 6, 8]
        first_iteration = False

    # Apply median filter based on the chi2 mean to obtain a smoother model
    print('=== BLOCK 2: Median-filtered output ===')
    Imodel = meu.apply_median_filter_based_on_chi2(
        Imodel, masked_chi2_mean, median_filter_chi2_mean_thres, median_filter_size)
    if verbose:
        iu.plot_inversion_output(Imodel, ll.mask, scale=scale, save_fig=False)
        iu.plot_mag(Imodel, ll.mask, scale=scale, save_fig=False)
    # init_model_from_sequence = True # Set the flag to True after the first iteration to avoid reinitialising the model

    # Run the ME inversion again based on the smoothed model input
    print('=== BLOCK 3: Randomised ME Inversions ===')
    Imodel, syn, chi2 = meu.run_randomised_me_inversion(
        Imodel, me, obs, sig, nRandom=nRandom2, nIter=nIter2, chi2_thres=chi2_thres2, mu=mu, verbose=verbose)
    masked_chi2_mean = iu.masked_mean(chi2, ll.mask)
    if verbose:
        print(f'Masked chi2 mean: {masked_chi2_mean:.2f}')
        iu.plot_inversion_output(Imodel, ll.mask, scale=scale, save_fig=False)
        iu.plot_mag(Imodel, ll.mask, scale=scale, save_fig=False)

    # Run the spatially regularised ME inversion
    print('=== BLOCK 4: Spatially Regularised ME Inversions ===')
    Imodel, syn, chi2 = meu.run_spatially_regularized_inversion(
        me, Imodel, obs, sig, nIter3, chi2_thres3, mu, alpha_strength, alpha_list, method=1, delay_bracket=3,
        dtype=dtype, verbose=True)
    Imodel = np.squeeze(Imodel)
    errors = me.estimate_uncertainties(Imodel, obs, sig, mu=mu)

    # Apply cavity error correction to the model
    # print('=== Cavity Error Correction ===')
    corrected_mo = meu.correct_velocities_for_cavity_error(Imodel, ll.cmap, l0, global_offset=0.0)
    if verbose:
        iu.plot_inversion_output(corrected_mo, ll.mask, scale=scale, save_fig=True,
                                 save_dir=save_dir, figname=f'inversion_output_{tt}.pdf')
        iu.plot_mag(corrected_mo, ll.mask, scale=scale, save_fig=True,
                    save_dir=save_dir, figname=f'mag_output_{tt}.pdf')
    else:
        iu.plot_inversion_output(corrected_mo, ll.mask, scale=scale, save_fig=True,
                                 save_dir=save_dir, figname=f'inversion_output_{tt}.pdf', show_fig=False)
        iu.plot_mag(corrected_mo, ll.mask, scale=scale, save_fig=True,
                    save_dir=save_dir, figname=f'mag_output_{tt}.pdf', show_fig=False)

    # Apply a mask to the model and errors to remove the NaN values from the edges
    print('=== BLOCK 6: Apply Mask to Model and Errors ===')
    masked_model = meu.apply_mask_to_model(corrected_mo, ll.mask, nan_mask_replacements)
    masked_errors = meu.apply_mask_to_model(errors, ll.mask, nan_mask_replacements)
    if verbose:
        print(f'Masked chi2 mean: {masked_chi2_mean:.2f}')
        iu.plot_inversion_output(masked_model, mask=None, scale=scale, save_fig=False)
        iu.plot_inversion_output(masked_errors, mask=None, scale=scale, save_fig=False,
                                 apply_median_filter=True, filter_index=[1, 2], filter_size=3)
        iu.plot_mag(masked_model, mask=None, scale=scale, save_fig=False)

    print('=== Calculating Blos and Bhor ===')
    # Rearrange the model and errors for saving
    model_im = rearrange(masked_model, 'ny nx nparams -> nparams ny nx')
    errors_im = rearrange(masked_errors, 'ny nx nparams -> nparams ny nx')

    # Create arrays with uncertainties
    B_with_errors = unumpy.uarray(model_im[0], errors_im[0])
    inc_with_errors = unumpy.uarray(model_im[1], errors_im[1])

    # Calculate Blos and Bhor with propagated errors
    Blos_with_errors = B_with_errors * unumpy.cos(inc_with_errors)
    Bhor_with_errors = B_with_errors * unumpy.sin(inc_with_errors)

    # Extract nominal values and standard deviations
    Blos = unumpy.nominal_values(Blos_with_errors)
    Blos_err = unumpy.std_devs(Blos_with_errors)

    Bhor = unumpy.nominal_values(Bhor_with_errors)
    Bhor_err = unumpy.std_devs(Bhor_with_errors)

    # Clip the errors to avoid very large values
    Bhor_err_clipped = np.clip(Bhor_err, a_min=0, a_max=np.max(Bhor))
    Blos_err_clipped = np.clip(Blos_err, a_min=0, a_max=np.max(np.abs(Blos)))

    # Extend model_im by adding Blos and Bhor and 10 and 11 indices and Mask as 12
    model_im = np.concatenate((model_im, Blos[np.newaxis], Bhor[np.newaxis], ll.mask[np.newaxis]), axis=0)
    errors_im = np.concatenate(
        (errors_im, Blos_err_clipped[np.newaxis], Bhor_err_clipped[np.newaxis], ll.mask[np.newaxis]), axis=0)

    print('=== Saving the Inversion Output ===')
    # Rearrange the model and errors for saving in IDL like format
    idl_model_im = rearrange(model_im, 'nparams ny nx -> nparams nx ny')
    idl_errors_im = rearrange(errors_im, 'nparams ny nx -> nparams nx ny')

    # Save the inversion output in the fits format
    time_string = all_start_times[tt].replace(':', '').replace(' ', '_T')
    cen_wav = str(int(central_wavelength))

    # Save the inversion output in the fits format
    for var in inversion_save_list:
        var_index = inversion_out_list.index(var)
        sav_var = inversion_out_list[var_index]
        out_file_name = save_dir + f'temp_{sav_var}_{cen_wav}_t_{tt}_{time_string}.fits'
        iu.save_fits(idl_model_im[var_index], fits_header, out_file_name, overwrite=True, verbose=verbose)
        if inversion_save_errors_fits and sav_var != 'Nan_mask':
            out_file_name = save_dir + f'temp_{sav_var}_err_{cen_wav}_t_{tt}_{time_string}.fits'
            iu.save_fits(idl_errors_im[var_index], fits_header, out_file_name, overwrite=True, verbose=verbose)

# %%
time_index_range = f"{time_range[0]}-{time_range[-1]}"
obs_start_time = all_start_times[time_range[0]].replace(':', '').replace(' ', '_T')
obs_end_time = all_start_times[time_range[-1]].replace(':', '').replace(' ', '_T')
for var in inversion_save_list:
    var_index = inversion_out_list.index(var)
    sav_var = inversion_out_list[var_index]
    temp_file_list = []
    # create a numpy array with nx, ny , nt dimensions
    full_var_data = np.zeros((nt, nx, ny))

    if inversion_save_errors_fits or inversion_save_errors_lp:
        full_err_data = np.zeros((nt, nx, ny))

    for ii in range(len(time_range)):
        tt = time_range[ii]
        time_string = all_start_times[tt].replace(':', '').replace(' ', '_T')
        out_file_name = save_dir + f'temp_{sav_var}_{cen_wav}_t_{tt}_{time_string}.fits'
        var_data = iu.load_fits_data(out_file_name)
        temp_file_list.append(out_file_name)
        var_header = iu.load_fits_header(out_file_name, out_dict=False)
        full_var_data[ii] = var_data

        if inversion_save_errors_fits and sav_var != 'Nan_mask':
            err_out_file_name = save_dir + f'temp_{sav_var}_err_{cen_wav}_t_{tt}_{time_string}.fits'
            err_data = iu.load_fits_data(err_out_file_name)
            temp_file_list.append(err_out_file_name)
            full_err_data[ii] = err_data

    if var in inversion_save_fits_list:
        # Save the full variable data
        out_file_name = save_dir + f'{sav_var}_{cen_wav}_{obs_start_time}_{obs_end_time}_t_{time_index_range}.fits'
        iu.save_fits(full_var_data, var_header, out_file_name, overwrite=True, verbose=verbose)
        if inversion_save_errors_fits and sav_var != 'Nan_mask':
            err_out_file_name = save_dir + \
                f'{sav_var}_err_{cen_wav}_{obs_start_time}_{obs_end_time}_t_{time_index_range}.fits'
            iu.save_fits(full_err_data, var_header, err_out_file_name, overwrite=True, verbose=verbose)

    if var in inversion_save_lp_list:
        # Save the inversion output in the LP format
        lp_out_file_name = save_dir + f'{sav_var}_{cen_wav}_{obs_start_time}_{obs_end_time}_t_{time_index_range}.fcube'
        lp_data = np.float32(rearrange(full_var_data, 'nt nx ny -> nx ny nt'))
        lp.writeto(lp_out_file_name, lp_data, extraheader='', dtype=None, verbose=verbose, append=False)

        if inversion_save_errors_lp and sav_var != 'Nan_mask':
            lp_err_out_file_name = save_dir + \
                f'{sav_var}_err_{cen_wav}_{obs_start_time}_{obs_end_time}_nt_{time_index_range}.fcube'
            lp_err_data = np.float32(rearrange(full_err_data, 'nt nx ny -> nx ny nt'))
            lp.writeto(lp_err_out_file_name, lp_err_data, extraheader='', dtype=None, verbose=verbose, append=False)

    # Delete the temporary file using os module
    for temp_file in temp_file_list:
        if verbose:
            print(f'Deleting temporary file: {temp_file}')
        os.remove(temp_file)

# %%

print('=== Save the Run Config ===')
# combine input_config and inversion_config dictionaries
full_config = {**input_config, **inversion_config}

# Save the input and inversion configuration as a separate file
iu.save_yaml_config(full_config, 'full_config.yaml', save_dir=save_dir)
# Save the fits information as a separate file
iu.save_yaml_config(fits_info, 'fits_info.yaml', save_dir=save_dir, append_timestamp=False)

# Save the fits header as a separate file

iu.save_fits_header_as_text(fits_header_dict, 'fits_header.txt', save_dir=save_dir)

# %% [markdown]
# ---

# %%
# datadir = '/mn/stornext/d18/lapalma/reduc/2020/2020-08-07/CRISP/cubes_nb/'
# blos_cube = datadir + 'Blos.6173_2020-08-07T08:22:14.icube'
# bhor_cube = datadir + 'Bhor.6173_2020-08-07T08:22:14.icube'

# data_dir = '/mn/stornext/d18/lapalma/reduc/2021/2021-06-22/CRISP/cubes_nb/'
# blos_old = data_dir + 'Blos.6173_2021-06-22T08:17:48.fcube'
# bhor_old = data_dir + 'Bhor.6173_2021-06-22T08:17:48.fcube'

# %%
# blos_new = 'temp/Blos_6173_2021-06-22_T090257_2021-06-22_T090257_t_145-145.fcube'
# bhor_new = 'temp/Bhor_6173_2021-06-22_T090257_2021-06-22_T090257_t_145-145.fcube'

# %%
# iu.plot_sst_blos_bhor(blos_new, bhor_new, tt=0,xrange=xrange, yrange=yrange, figsize=(20,10), fontsize=12, vmin1=-50,
#  vmax1=50, vmax2=200)
# iu.plot_sst_blos_bhor(blos_old, bhor_old, tt=145,xrange=xrange, yrange=yrange, figsize=(20,10), fontsize=12,
#  crop=crop, vmin1=-50, vmax1=50, vmax2=200)
