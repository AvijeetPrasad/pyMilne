import subprocess
import os
import numpy as np


def write_ambig_input(outdir, pix, pbr, lonlat, blos, bhor, bazi, verbose=False):
    """
    Write the inputs for the ambiguity resolution to a dat file.

    Parameters:
    nx (int): Number of pixels along x-axis.
    ny (int): Number of pixels along y-axis.
    outdir (str): Output directory.
    pix (tuple): Pixel size (xpix, ypix).
    pbr (tuple): Parameters b, p, and radius.
    lonlat (tuple): Longitude and latitude (theta, phi).
    blos (np.ndarray): 2D array for blos.
    bhor (np.ndarray): 2D array for bhor.
    bazi (np.ndarray): 2D array for bazi.
    """
    ny, nx = blos.shape
    xpix = pix
    ypix = pix
    b, p, radius = pbr
    theta, phi = lonlat

    # outfile = f'{outdir}/input.dat'
    outfile = os.path.join(outdir, 'ambig_input.dat')

    with open(outfile, 'w') as file:
        # Write header information with adjusted formatting
        file.write(f'{nx} {ny} nx ny\n')
        file.write(f'{xpix:.6f} {ypix:.6f} xpix ypix\n')
        file.write(f'{b:.6f} {p:.6f} {radius:.6f} b p radius\n')
        file.write(f'{theta:.6f} {phi:.6f} theta phi\n')

        # Formatting string for the field data
        fmtstr = ' '.join(['{:13.6f}'] * nx) + '\n'

        # Write the blos, bhor, and bazi arrays
        for j in range(ny):
            file.write(fmtstr.format(*blos[j]))
        for j in range(ny):
            file.write(fmtstr.format(*bhor[j]))
        for j in range(ny):
            file.write(fmtstr.format(*bazi[j]))
    if verbose:
        print(f'Output file: {outfile}')
    return outfile


def run_command(command, verbose=True):
    """
    Run a shell command and print the output in real-time.

    Parameters:
    command (str): The shell command to run.

    Example usage:
    # run_command('/path/to/executable/ambig parameters.par')
    """
    if verbose:
        print("Command:", command)

    # Run the command using subprocess
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read the stdout and stderr in real time
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())

    # Capture any remaining error output
    stderr = process.stderr.read()
    if stderr:
        print("Errors:\n", stderr.strip())

    # Check the return code
    return_code = process.poll()
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
    else:
        print("Command completed successfully")


def read_ambig_dat_file(file_path, new_shape, verbose=False):
    """
    Reads a 2D array from a Fortran-written dat file into a NumPy array and reshapes it.

    Parameters:
    file_path (str): Path to the dat file.
    new_shape (tuple): New shape for the array (rows, columns).

    Returns:
    np.ndarray: Reshaped 2D NumPy array containing the data from the file.
    """
    if verbose:
        print(f"Reading file: {file_path}")
    with open(file_path, 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

    # Process each line to extract the floating-point numbers
    data = []
    for line in lines:
        # Split the line by whitespace and convert each part to a float
        row = [float(value) for value in line.split()]
        data.append(row)

    # Convert the list of lists into a NumPy array
    array = np.array(data)

    # Flatten the array and then reshape it to the desired shape
    ambig = array.flatten().reshape(new_shape)
    if verbose:
        print(f"Array shape: {ambig.shape}")
    return ambig
