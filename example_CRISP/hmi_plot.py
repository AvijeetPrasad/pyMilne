import os
from sunpy.net import Fido, attrs as a
import astropy.units as u
import sunpy.map
import astropy.io.fits
import astropy.wcs
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame


def fetch_and_prepare_data(series, email, time_start, path='temp/'):
    res = Fido.search(a.Time(time_start, time_start), a.jsoc.Series(series), a.jsoc.Notify(email))
    # check if the file already exists else download it
    time_str = res['jsoc'][0]['T_REC'].replace(':', '').replace('.', '')
    if series == 'hmi.Ic_45s':
        segment = 'continuum'
    elif series == 'hmi.M_45s':
        segment = 'magnetogram'
    file_path = f'{path}{series.casefold()}.{time_str}.2.{segment}.fits'
    if not os.path.exists(file_path):
        files = Fido.fetch(res, path=path, overwrite=False)
        print(f'File downloaded: {files}')
        file_path = files
    else:
        print(f'File already exists: {file_path}')
    return file_path


def read_fits(file):
    print(f'Reading file: {file}')
    with astropy.io.fits.open(file, mode='readonly') as hdul:
        data = hdul[1].data
        header = hdul[1].header
    wcs = astropy.wcs.WCS(header=header)
    return sunpy.map.Map(data, wcs).rotate()


def plot_submaps(map1, map2, bottom_left, top_right, center_coord=None, draw_circle=False, radius=87,
                 draw_rectangle=False, height=56, width=56, rot_fov=0):
    submap1 = map1.submap(bottom_left, top_right=top_right)
    submap2 = map2.submap(bottom_left, top_right=top_right)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': submap1})
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


def plot_hmi_ic_mag(tstart, ic_series, mag_series, email, x1, x2, y1, y2, draw_circle=False, radius=87,
                    draw_rectangle=False, height=56, width=56, rot_fov=0, save_dir='temp/'):

    ic_file = fetch_and_prepare_data(ic_series, email, tstart, path=save_dir)
    blos_file = fetch_and_prepare_data(mag_series, email, tstart, path=save_dir)

    ic_map = read_fits(ic_file)
    blos_map = read_fits(blos_file)

    center_coord = SkyCoord((x1 + x2) / 2 * u.arcsec, (y1 + y2) / 2 * u.arcsec, frame=ic_map.coordinate_frame)
    top_right = SkyCoord(x2 * u.arcsec, y2 * u.arcsec, frame=ic_map.coordinate_frame)
    bottom_left = SkyCoord(x1 * u.arcsec, y1 * u.arcsec, frame=ic_map.coordinate_frame)

    plot_submaps(ic_map, blos_map, bottom_left, top_right, center_coord,
                 draw_circle, radius, draw_rectangle, height, width, rot_fov)


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
