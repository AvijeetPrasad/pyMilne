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


def read_azimuth_dat_file(file_path, new_shape, verbose=False):
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


def read_ambig_par_file(par_file):
    """
    Parses the input parameter file and returns the parameters as a dictionary.

    Parameters:
    - par_file (str): Path to the parameter file.

    Returns:
    - dict: Dictionary containing the parsed parameters.
    """
    params = {}

    with open(par_file, 'r') as f:
        # Read the filename line
        params['filename'] = f.readline().strip()

        # Read the flags line
        flags = f.readline().strip().split()
        params['irflag'] = int(flags[0])
        params['ibflag'] = int(flags[1])
        params['iaflag'] = int(flags[2])
        params['igflag'] = int(flags[3])
        params['ipflag'] = int(flags[4])

        # Read the npad, nap, ntx, nty line
        npad_nap_ntx_nty = f.readline().strip().split()
        params['npad'] = int(npad_nap_ntx_nty[0])
        params['nap'] = int(npad_nap_ntx_nty[1])
        params['ntx'] = int(npad_nap_ntx_nty[2])
        params['nty'] = int(npad_nap_ntx_nty[3])

        # Read the athresh, bthresh, nerode, ngrow line
        athresh_bthresh_nerode_ngrow = f.readline().strip().split()
        params['athresh'] = float(athresh_bthresh_nerode_ngrow[0])
        params['bthresh'] = float(athresh_bthresh_nerode_ngrow[1])
        params['nerode'] = int(athresh_bthresh_nerode_ngrow[2])
        params['ngrow'] = int(athresh_bthresh_nerode_ngrow[3])

        # Read the iaunit, ipunit, incflag line
        iaunit_ipunit_incflag = f.readline().strip().split()
        params['iaunit'] = int(iaunit_ipunit_incflag[0])
        params['ipunit'] = int(iaunit_ipunit_incflag[1])
        params['incflag'] = int(iaunit_ipunit_incflag[2])

        # Read the iseed, iverb, neq line
        iseed_iverb_neq = f.readline().strip().split()
        params['iseed'] = int(iseed_iverb_neq[0])
        params['iverb'] = int(iseed_iverb_neq[1])
        params['neq'] = int(iseed_iverb_neq[2])

        # Read the lambda, tfac0, tfactr line
        lambda_tfac0_tfactr = f.readline().strip().split()
        params['lambda'] = float(lambda_tfac0_tfactr[0])
        params['tfac0'] = float(lambda_tfac0_tfactr[1])
        params['tfactr'] = float(lambda_tfac0_tfactr[2])

        # Check if additional FITS or Hinode SOT/SP pipeline information is needed
        if params['irflag'] == 1:
            fits_info = f.readline().strip().split()
            params['iblong'] = int(fits_info[0])
            params['ibtrans'] = int(fits_info[1])
            params['ibazim'] = int(fits_info[2])
            params['ibfill'] = int(fits_info[3])

            params['cmdkey'] = f.readline().strip()
            params['latkey'] = f.readline().strip()
            params['xcenkey'] = f.readline().strip()
            params['ycenkey'] = f.readline().strip()
            params['pkey'] = f.readline().strip()
            params['bkey'] = f.readline().strip()
            params['radkey'] = f.readline().strip()
            params['pxkey'] = f.readline().strip()
            params['pykey'] = f.readline().strip()

        elif params['irflag'] == 2:
            hinode_info = f.readline().strip().split()
            params['jfs'] = int(hinode_info[0])
            params['jfi'] = int(hinode_info[1])
            params['jfa'] = int(hinode_info[2])
            params['jsf'] = int(hinode_info[3])
            params['jci'] = int(hinode_info[4])

    return params


def write_ambig_par_file(par_file, params, labels=None, pprint=False):
    """
    Writes the parameter values from a dictionary to the parameter file with proper spacing.

    Parameters:
    - par_file (str): Path to the parameter file.
    - params (dict): Dictionary containing the parameters.
    - labels (list): List of labels for each line in the parameter file.
    """
    if labels is None:
        # Define the labels for each line
        labels = [
            "irflag   ibflag   iaflag   igflag   ipflag",
            "npad     nap      ntx      nty",
            "athresh  bthresh  nerode   ngrow",
            "iaunit   ipunit   incflag",
            "iseed    iverb    neq",
            "lambda   tfac0    tfactr"
        ]

    # Determine the maximum length of the labels section for proper alignment
    max_label_length = max(len(label) for label in labels)

    with open(par_file, 'w') as f:
        # Write the filename line
        f.write(f"{params['filename']}\n")

        # Write the flags line
        line = f"{params['irflag']:<8}{params['ibflag']:<8}{
            params['iaflag']:<8}{params['igflag']:<8}{params['ipflag']:<8}"
        f.write(f"{line}{' ' * (max_label_length - len(line) + 4)}{labels[0]}\n")

        # Write the npad, nap, ntx, nty line
        line = f"{params['npad']:<8}{params['nap']:<8}{params['ntx']:<8}{params['nty']:<8}"
        f.write(f"{line}{' ' * (max_label_length - len(line) + 4)}{labels[1]}\n")

        # Write the athresh, bthresh, nerode, ngrow line
        line = f"{params['athresh']:<8}{params['bthresh']:<8}{params['nerode']:<8}{params['ngrow']:<8}"
        f.write(f"{line}{' ' * (max_label_length - len(line) + 4)}{labels[2]}\n")

        # Write the iaunit, ipunit, incflag line
        line = f"{params['iaunit']:<8}{params['ipunit']:<8}{params['incflag']:<8}"
        f.write(f"{line}{' ' * (max_label_length - len(line) + 4)}{labels[3]}\n")

        # Write the iseed, iverb, neq line
        line = f"{params['iseed']:<8}{params['iverb']:<8}{params['neq']:<8}"
        f.write(f"{line}{' ' * (max_label_length - len(line) + 4)}{labels[4]}\n")

        # Write the lambda, tfac0, tfactr line
        line = f"{params['lambda']:<8}{params['tfac0']:<8}{params['tfactr']:<8}"
        f.write(f"{line}{' ' * (max_label_length - len(line) + 4)}{labels[5]}\n")

        # Write additional FITS or Hinode SOT/SP pipeline information if needed
        if params['irflag'] == 1:
            f.write(f"{params['iblong']:<8}{params['ibtrans']:<8}{params['ibazim']:<8}{params['ibfill']:<8}\n")
            f.write(f"{params['cmdkey']:<8}\n")
            f.write(f"{params['latkey']:<8}\n")
            f.write(f"{params['xcenkey']:<8}\n")
            f.write(f"{params['ycenkey']:<8}\n")
            f.write(f"{params['pkey']:<8}\n")
            f.write(f"{params['bkey']:<8}\n")
            f.write(f"{params['radkey']:<8}\n")
            f.write(f"{params['pxkey']:<8}\n")
            f.write(f"{params['pykey']:<8}\n")

        elif params['irflag'] == 2:
            f.write(f"{params['jfs']:<8}{params['jfi']:<8}{params['jfa']:<8}{params['jsf']:<8}{params['jci']:<8}\n")
        if pprint:
            print(f"Parameter file: {par_file}")
            for key, value in params.items():
                print(f"{key}: {value}")
    return par_file


def get_par_info(param_name, verbose=False):
    """
    Returns the help text for the given parameter.

    Parameters:
    - param_name (str): The name of the parameter.

    Returns:
    - str: Help text for the parameter.
    """
    help_texts = {
        'irflag': (
            "irflag determines the format of the input data:\n"
            "- 0: Formatted text file\n"
            "- 1: Generic FITS format\n"
            "- 2: FITS format with extensions found in the Hinode SOT/SP pipeline"
        ),
        'ibflag': (
            "ibflag determines how the field is specified:\n"
            "- 0: Line of sight and transverse components plus the azimuthal angle\n"
            "- 1: Magnitude of the field plus the inclination and azimuthal angles"
        ),
        'iaflag': (
            "iaflag determines the direction of zero azimuthal angle:\n"
            "- 0: CCD+x\n"
            "- 1: CCD+y\n"
            "- 2: CCD-x\n"
            "- 3: CCD-y"
        ),
        'igflag': (
            "igflag determines the geometry to use:\n"
            "- 1: Planar geometry (faster for smaller fields of view)\n"
            "- 2: Spherical geometry (better for larger fields of view covering a\
                significant fraction of the solar disk)"
        ),
        'ipflag': (
            "ipflag determines whether to perform a potential field acute angle ambiguity resolution:\n"
            "- 0: Do not perform\n"
            "- 1: Perform ambiguity resolution"
        ),
        'npad': (
            "npad determines the number of pixels to add on each side of the field of view\
                  to mitigate the effects of periodic boundary conditions in FFTs.\n"
            "Recommended to set npad to approximately 10% of the number of pixels\
                  across the disk for full disk images.\n"
            "Recommended Range: 0 to 200"
        ),
        'nap': (
            "nap determines the number of pixels over which to smoothly transition the field to zero\
                  outside the field of view (planar geometry), "
            "or number of pixels outside a tile to include in the potential field calculation (spherical geometry).\n"
            "It must be less than npad else will be reset to equal npad.\n"
            "Recommended Range: 0 to 50"
        ),
        'ntx': (
            "ntx sets the number of tiles in the x-direction (longitude) used to construct\
                  the potential field derivatives when using spherical geometry.\n"
            "Recommended Range: 0 to 50"
        ),
        'nty': (
            "nty sets the number of tiles in the y-direction (latitude) used to construct\
                  the potential field derivatives when using spherical geometry.\n"
            "Recommended Range: 0 to 50"
        ),
        'athresh': (
            "athresh is the transverse field strength above which annealing will be used.\n"
            "Typical values: 200G to 400G, but can be set to 0 if speed is not a concern.\n"
            "Recommended Range: 0 to 400"
        ),
        'bthresh': (
            "bthresh is the transverse field strength below which the\
                  nearest neighbour acute angle method will be used.\n"
            "Typical values: 200G to 400G, but should be set higher than athresh.\n"
            "Recommended Range: 0 to 500"
        ),
        'nerode': (
            "nerode is the number of pixels by which to erode the mask to prevent the inclusion of\
                  isolated above-threshold pixels in the annealing.\n"
            "Typical value: 1\n"
            "Recommended Range: 1 to 5"
        ),
        'ngrow': (
            "ngrow is the number of pixels by which to subsequently grow the mask to allow a\
                  buffer around above-threshold pixels in the annealing.\n"
            "Typical value: 1\n"
            "Recommended Range: 1 to 5"
        ),
        'iaunit': (
            "iaunit specifies whether azimuth is measured in radians or degrees:\n"
            "- 0: Radians\n"
            "- 1: Degrees"
        ),
        'ipunit': (
            "ipunit specifies whether pointing is measured in radians or degrees:\n"
            "- 0: Radians\n"
            "- 1: Degrees"
        ),
        'incflag': (
            "incflag specifies whether an inclination of 0 is vertical or horizontal:\n"
            "- 0: Vertical\n"
            "- 1: Horizontal"
        ),
        'iseed': (
            "iseed is used for initializing the random number generator.\n"
            "Must be a positive integer."
            "Recommended Range: 1 to 20"
        ),
        'iverb': (
            "iverb controls the verbosity of the output:\n"
            "- 0: Suppresses all print statements\n"
            "- 1: Prints a statement when each stage of the code finishes\n"
            "- 2: Prints the energy at each temperature in the annealing, plus number of pixels\
                  flipped during each iteration in smoothing (produces a large amount of output)"
        ),
        'neq': (
            "neq determines the number of times pixels are visited at a given temperature.\n"
            "Must be a positive integer.\n"
            "Recommened Range: 0 to 150"
        ),
        'lambda': (
            "lambda is the weighting factor between the divergence term and the vertical current density term.\n"
            "Must be positive (or zero).\n"
            "Recommened Range: 0 to 1"
        ),
        'tfac0': (
            "tfac0 scales the initial temperature.\n"
            "Must be positive and typically should not be smaller than of order 1.\n"
            "Recommened Range: 1.0 to 3.0"
        ),
        'tfactr': (
            "tfactr determines the cooling rate.\n"
            "Must have 0 < tfactr < 1, with slower cooling for values closer to 1.\n"
            "Slower cooling is more likely to find the global minimum, but takes longer to run.\n"
            "Range: 0 to 1"
        )
    }
    help_text = help_texts.get(param_name, f"No help text available for parameter: {param_name}")
    if verbose:
        print(help_text)
    return help_text


def get_par_range(param_name, start=None, end=None, step=None, verbose=False):
    """
    Returns the acceptable range of values for the given parameter.

    Parameters:
    - param_name (str): The name of the parameter.
    - start (float, optional): The start value for continuous ranges.
    - end (float, optional): The end value for continuous ranges.
    - step (float, optional): The step size for continuous ranges.
    - verbose (bool, optional): If True, prints the help text for the parameter.

    Returns:
    - list: List of acceptable values for discrete parameters.
    - numpy.ndarray: Array of acceptable values for continuous parameters.
    """
    if verbose:
        print(get_par_info(param_name))

    discrete_params = {
        'irflag': [0, 1, 2],
        'ibflag': [0, 1],
        'iaflag': [0, 1, 2, 3],
        'igflag': [1, 2],
        'ipflag': [0, 1],
        'iaunit': [0, 1],
        'ipunit': [0, 1],
        'incflag': [0, 1],
        'iseed': range(1, 20),  # assuming positive integers
        'iverb': [0, 1, 2],
    }

    ranged_params = {
        'athresh': (0, 400),
        'bthresh': (0, 500),
        'lambda': (0, 1),
        'tfac0': (1.0, 3.0),
        'tfactr': (0, 1),
        'npad': (0, 200),
        'nap': (0, 50),
        'ntx': (0, 50),
        'nty': (0, 50),
        'nerode': (1, 5),
        'ngrow': (1, 5),
        'neq': (0, 150),
    }

    if param_name in discrete_params:
        return discrete_params[param_name]
    elif param_name in ranged_params:
        if start is None:
            start = ranged_params[param_name][0]
        if end is None:
            end = ranged_params[param_name][1]
        if step is None:
            # Generate three points: min, max, and middle
            return np.array([start, (start + end) / 2, end])
        else:
            return np.arange(start, end + step, step)
    else:
        raise ValueError(f"Unknown parameter name: {param_name}")
