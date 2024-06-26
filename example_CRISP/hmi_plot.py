import os
from sunpy.net import Fido, attrs as a
import astropy.units as u
import sunpy.map
import astropy.io.fits
import astropy.wcs
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from enhance_data import run_enhance
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sunpy.time import TimeRange
# from matplotlib import cm
# WARNING: FITSFixedWarning: 'datfix' made the change 'Set MJD-OBS to 59068.348353 from DATE-OBS'. [astropy.wcs.wcs]
import warnings
warnings.filterwarnings('ignore', category=astropy.wcs.FITSFixedWarning)


def fetch_and_prepare_data_vso_client(series, time_start, path='temp/', debug=False):
    # add a buffer of 45 seconds to the time_start to ensure the file is downloaded
    time_range = TimeRange(time_start, 45 * u.second)
    time_end = time_range.end.iso
    if series == 'hmi.Ic_45s':
        segment = 'continuum'
        res = Fido.search(a.Time(time_start, time_end), a.Instrument.hmi, a.Physobs.intensity)
    elif series == 'hmi.M_45s':
        segment = 'magnetogram'
        res = Fido.search(a.Time(time_start, time_end), a.Instrument.hmi, a.Physobs.los_magnetic_field)
    if debug:
        print(f'Query result: {res}')
    series = series.replace('.', '_')
    time_str = res['vso'][0]['Start Time'].iso.replace('-', '_').replace(':', '_').replace(' ', '_')[:-4] + '_tai'
    file_path_expected = f'{path}{series.casefold()}_{time_str}_{segment}.fits'
    if not os.path.exists(file_path_expected):
        files = Fido.fetch(res['vso'][0], path=path, overwrite=False)
        if debug:
            print(f'File downloaded: {files}')
        file_path = files[0]
        # rename the file as per the expected name
        os.rename(file_path, file_path_expected)  # file is downloaded with a slightly different timestamp (why?)
    else:
        print(f'File already exists: {file_path_expected}')
    return file_path_expected


def fetch_and_prepare_data(series, email, time_start, path='temp/', debug=False):
    res = Fido.search(a.Time(time_start, time_start), a.jsoc.Series(series), a.jsoc.Notify(email))
    if debug:
        print(f'Query result: {res}')
    # check if the file already exists else download it
    time_str = res['jsoc'][0]['T_REC'].replace(':', '').replace('.', '')
    if series == 'hmi.Ic_45s':
        segment = 'continuum'
    elif series == 'hmi.M_45s':
        segment = 'magnetogram'
    file_path = f'{path}{series.casefold()}.{time_str}.2.{segment}.fits'
    if not os.path.exists(file_path):
        files = Fido.fetch(res, path=path, overwrite=False)
        if debug:
            print(f'File downloaded: {files}')
        file_path = files[0]
    else:
        print(f'File already exists: {file_path}')
    return file_path


def read_fits(file):
    print(f'Reading file: {file}')
    with astropy.io.fits.open(file, mode='readonly') as hdul:
        try:
            data = hdul[1].data
            header = hdul[1].header
        except IndexError:
            data = hdul[0].data
            header = hdul[0].header
    wcs = astropy.wcs.WCS(header=header)
    return sunpy.map.Map(data, wcs).rotate()


def plot_submaps(map1, map2, bottom_left, top_right, center_coord, shape, rot_fov=0, figsize=(16, 8)):

    if shape[0] == 'circle':
        draw_rectangle = False
        draw_circle = True
        radius = shape[1]
    else:
        draw_rectangle = True
        draw_circle = False
        width = shape[1]
        height = shape[2]

    submap1 = map1.submap(bottom_left, top_right=top_right)
    submap2 = map2.submap(bottom_left, top_right=top_right)

    fig, axes = plt.subplots(1, 2, figsize=figsize, subplot_kw={'projection': submap1})
    # set axes[1] subplot_kw projection based on submap2
    # Set the projection for the second subplot
    axes[1].remove()  # Remove the existing axis
    ax2 = fig.add_subplot(1, 2, 2, projection=submap2)
    axes[1] = ax2  # Reassign the axis to the axes array
    if draw_circle:
        radius = radius * u.arcsec
        # Drawing rectangles rotated 1 degree apart to form a circle
        for angle in np.arange(0, 360, 0.5):
            rotation_angle = angle * u.deg
            offset_frame = SkyOffsetFrame(origin=center_coord, rotation=rotation_angle)
            rectangle = SkyCoord(lon=[-1/2, 1/2] * radius, lat=[-1/2, 1/2] * radius, frame=offset_frame)

            submap1.draw_quadrangle(
                rectangle,
                axes=axes[0],
                edgecolor="white",
                linestyle="--",
                linewidth=1,
            )
            submap2.draw_quadrangle(
                rectangle,
                axes=axes[1],
                edgecolor="white",
                linestyle="--",
                linewidth=1,
            )
    if draw_rectangle:
        width = width * u.arcsec
        height = height * u.arcsec
        rotation_angle = rot_fov * u.deg
        offset_frame = SkyOffsetFrame(origin=center_coord, rotation=rotation_angle)
        rectangle = SkyCoord(lon=[-1/2, 1/2] * width, lat=[-1/2, 1/2] * height, frame=offset_frame)

        submap1.draw_quadrangle(
            rectangle,
            axes=axes[0],
            edgecolor="red",
            linestyle="--",
            linewidth=1,
        )
        submap2.draw_quadrangle(
            rectangle,
            axes=axes[1],
            edgecolor="red",
            linestyle="--",
            linewidth=1,
        )
    title_pad = 5
    submap1.plot(axes=axes[0])
    axes[0].set_title(axes[0].get_title(), pad=title_pad)
    axes[0].grid(False)

    submap2.plot(axes=axes[1])
    axes[1].set_title(axes[1].get_title(), pad=title_pad)
    axes[1].grid(False)

    plt.tight_layout()
    plt.show()


def plot_hmi_ic_mag(tstart, ic_series, mag_series, email, x1, x2, y1, y2, save_dir='.', enhance_ic=False,
                    enhance_m=False, figsize=(16, 8), overwrite=False, is_north_up=True,
                    fov_angle=0, shape=['circle', 87], buffer=5):

    if is_north_up:
        rot_fov = 0
    else:
        rot_fov = fov_angle
        buffer = 15

    if len(email) == 0:
        ic_file = fetch_and_prepare_data_vso_client(ic_series, tstart, path=save_dir)
        blos_file = fetch_and_prepare_data_vso_client(mag_series, tstart, path=save_dir)
    else:
        ic_file = fetch_and_prepare_data(ic_series, email, tstart, path=save_dir)
        blos_file = fetch_and_prepare_data(mag_series, email, tstart, path=save_dir)

    if enhance_ic or enhance_m:
        python_path = "/mn/stornext/d9/data/avijeetp/envs/enhance/bin/python"
        script_dir = "/mn/stornext/u3/avijeetp/codes/enhance"
        # get the full folder path for ic_file
        input_dir = os.path.dirname(ic_file)
        if enhance_ic:
            # get full path for ic_file
            ic_file = os.path.abspath(ic_file)
            # get the filename of ic_file
            ic_file_base = os.path.basename(ic_file)
            ic_output_file = os.path.join(input_dir, "enhanced_" + ic_file_base)
            ic_output_file = os.path.join(input_dir, "enhanced_" + ic_file_base)
            ic_file = run_enhance(python_path, script_dir, ic_file, "intensity", ic_output_file, overwrite=overwrite)
        if enhance_m:
            # get full path for blos_file
            blos_file = os.path.abspath(blos_file)
            blos_file_base = os.path.basename(blos_file)
            blos_output_file = os.path.join(input_dir, "enhanced_" + blos_file_base)
            blos_file = run_enhance(python_path, script_dir, blos_file, "intensity",
                                    blos_output_file, overwrite=overwrite)

    ic_map = read_fits(ic_file)
    blos_map = read_fits(blos_file)

    # Add buffer to the coordinates
    x1 -= buffer
    x2 += buffer
    y1 -= buffer
    y2 += buffer

    center_coord = SkyCoord((x1 + x2) / 2 * u.arcsec, (y1 + y2) / 2 * u.arcsec, frame=ic_map.coordinate_frame)
    top_right = SkyCoord(x2 * u.arcsec, y2 * u.arcsec, frame=ic_map.coordinate_frame)
    bottom_left = SkyCoord(x1 * u.arcsec, y1 * u.arcsec, frame=ic_map.coordinate_frame)

    plot_submaps(ic_map, blos_map, bottom_left, top_right, center_coord, shape, rot_fov, figsize=figsize)


def plot_sst_pointings(tstart, ic_series, hplnt, hpltt, email,
                       figsize=(8, 8), save_dir='temp/'):

    if len(email) == 0:
        ic_file = fetch_and_prepare_data_vso_client(ic_series, tstart, path=save_dir)
    else:
        ic_file = fetch_and_prepare_data(ic_series, email, tstart, path=save_dir)

    ic_map = read_fits(ic_file)

    x1 = np.min(hplnt[:, 0])
    x2 = np.max(hplnt[:, 1])
    y1 = np.min(hpltt[:, 0])
    y2 = np.max(hpltt[:, 1])

    # center_coord = SkyCoord((x1 + x2) / 2 * u.arcsec, (y1 + y2) / 2 * u.arcsec, frame=ic_map.coordinate_frame)
    top_right = SkyCoord(x2 * u.arcsec, y2 * u.arcsec, frame=ic_map.coordinate_frame)
    bottom_left = SkyCoord(x1 * u.arcsec, y1 * u.arcsec, frame=ic_map.coordinate_frame)

    submap = ic_map.submap(bottom_left, top_right=top_right)

    fig, axes = plt.subplots(1, 1, figsize=figsize, subplot_kw={'projection': submap})

    norm = Normalize(vmin=0, vmax=len(hplnt) - 1)
    cmap = plt.get_cmap('Blues')

    for tt in range(len(hplnt)):
        x1t = hplnt[tt, 0]
        x2t = hplnt[tt, 1]
        y1t = hpltt[tt, 0]
        y2t = hpltt[tt, 1]
        width = x2t - x1t
        height = y2t - y1t
        width = width * u.arcsec
        height = height * u.arcsec
        rotation_angle = 0
        center_coordt = SkyCoord((x1t + x2t) / 2 * u.arcsec, (y1t + y2t) / 2 * u.arcsec, frame=ic_map.coordinate_frame)
        offset_frame = SkyOffsetFrame(origin=center_coordt, rotation=rotation_angle)
        rectangle = SkyCoord(lon=[-1/2, 1/2] * width, lat=[-1/2, 1/2] * height, frame=offset_frame)

        submap.draw_quadrangle(
            rectangle,
            axes=axes,
            edgecolor=cmap(norm(tt)),
            linestyle="--",
            linewidth=0.25,
        )

    title_pad = 5
    submap.plot(axes=axes)
    axes.set_title(axes.get_title(), pad=title_pad)
    axes.grid(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Dummy array for colorbar
    cbar = plt.colorbar(sm, ax=axes, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Time step')

    plt.tight_layout()
    plt.show()


def main():
    # Example usage
    tstart = '2024-05-21T10:19:05'
    ic_series = 'hmi.Ic_45s'
    mag_series = 'hmi.M_45s'
    email = 'avijeet.prasad@astro.uio.no'
    save_dir = 'temp/'
    x1 = -242.40
    x2 = -149.50
    y1 = -177.44
    y2 = -87.83
    plot_hmi_ic_mag(tstart, ic_series, mag_series, email, x1, x2, y1, y2, draw_rectangle=True,
                    rot_fov=0, width=56, height=56, draw_circle=False, radius=87, save_dir=save_dir)


if __name__ == "__main__":
    main()
