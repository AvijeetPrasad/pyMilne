import os
import numpy as np
from datetime import datetime as dt, timedelta
import argparse


def fov_angle(datetime_str=None, verbose=False):
    """
    Calculate the FOV angle based on the telescope turret log.

    Args:
    datetime_str (str): Optional. The datetime string in 'YYYY-MM-DDTHH:MM:SS.ssssss' or 'YYYY-MM-DDTHH:MM:SS' format.

    Returns:
    None
    """
    # Define the table constant
    TC = 48.0

    # If datetime_str is not provided, prompt the user for inputs
    if datetime_str is None:
        # Yesterday's date as default value
        yesterday_date = (dt.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        date_input = input(
            f"Enter the date (default: {yesterday_date}): ") or yesterday_date
        timestamp = input(
            "Enter the timestamp in HH:MM:SS format (default: 12:00:00): ") or "12:00:00"
        datetime_str = date_input + 'T' + timestamp

    # Handle cases where milliseconds are not provided
    if '.' not in datetime_str:
        datetime_str += '.000000'

    # Transform the datetime_str to a datetime object
    target_datetime = dt.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S.%f')

    # Extract date and time components
    target_date = target_datetime.strftime('%Y%m%d')
    target_time = target_datetime

    # Define the URL and the file to retrieve
    url = 'http://www.sst.iac.es/Logfiles/turret/'

    # Construct the file path based on the date
    file = target_date[:4] + '/positionLog_' + target_date[:4] + \
        '.' + target_date[4:6] + '.' + target_date[6:]
    final_url = url + file
    print(f'Source: {final_url}')

    # Retrieve the file if it does not exist
    file_name = file.split('/')[-1]
    if not os.path.isfile(file_name):
        os.system('wget -q ' + final_url)

    # Load the data
    data = np.loadtxt(file_name, skiprows=2, dtype='str')
    time_date = data[:, 0]
    time_time = data[:, 1]
    azimuth = data[:, 2].astype(float)
    elevation = data[:, 3].astype(float)
    tilt = data[:, -3].astype(float)

    # Combine date and time into datetime format
    time = [dt.strptime(time_date[i] + ' ' + time_time[i],
                        '%Y/%m/%d %H:%M:%S') for i in range(len(time_date))]

    # Find the index of the closest time to the target time
    time_diffs = np.abs([t - target_time for t in time])
    closest_time_index = np.argmin(time_diffs)

    # Get the corresponding azimuth, elevation, and tilt values
    phi = azimuth[closest_time_index]
    theta = elevation[closest_time_index]
    beta = tilt[closest_time_index]

    # Calculate the FOV angle
    fov_angle = phi - theta - TC - beta

    if verbose:
        # Print the results
        print(f'Requested time: {target_time.strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'Turret time: {time[closest_time_index].strftime("%Y-%m-%d %H:%M:%S")}')
        print(f'FOV angle: {fov_angle:.2f} degrees')

    # Delete the file
    os.system('rm ' + file_name)
    return fov_angle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the FOV angle based on the telescope turret log.",
        epilog="Example: python get_fov_angle.py --datetime '2021-06-22T08:43:09.52582'"
    )
    parser.add_argument('--datetime', type=str,
                        help="The datetime string in 'YYYY-MM-DDTHH:MM:SS.ssssss' or 'YYYY-MM-DDTHH:MM:SS' format.")

    args = parser.parse_args()

    if args.datetime:
        fov_angle(args.datetime, verbose=True)
    else:
        fov_angle(verbose=True)
