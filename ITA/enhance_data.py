import subprocess
import os
import tempfile
import warnings
import astropy.io.fits
# filter WARNING: VerifyWarning: Invalid 'BLANK' keyword in header.
warnings.filterwarnings('ignore', category=astropy.io.fits.verify.VerifyWarning)


def run_enhance(python_path, script_dir, input_file, file_type, output_file, debug_mode=False, overwrite=False):
    """
    Run the enhance.py script with the specified parameters.

    Parameters:
    - python_path (str): Path to the Python interpreter in the conda environment.
    - script_dir (str): Directory containing the enhance.py script.
    - input_file (str): Path to the input file.
    - file_type (str): Type of the file (e.g., 'intensity', 'blos').
    - output_file (str): Path to the output file.
    - debug_mode (bool): If True, print the command instead of running it.
    """
    # Check if the Python interpreter exists
    if not os.path.exists(python_path):
        print(f"Error: Python interpreter '{python_path}' does not exist.")
        return

    # Check if the script directory exists
    if not os.path.exists(script_dir):
        print(f"Error: Script directory '{script_dir}' does not exist.")
        return

    # Check if the input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return

    # Check if the output file exists and overwrite is False
    if os.path.exists(output_file) and not overwrite:
        print(f"Output file '{output_file}' already exists. Use overwrite=True to overwrite the file.")
        return output_file

    # Construct the shell script content
    script_content = f"""#!/bin/bash
cd {script_dir}
{python_path} enhance.py -i "{input_file}" -t "{file_type}" -o "{output_file}"
"""

    # Create a temporary file to hold the shell script
    with tempfile.NamedTemporaryFile('w', delete=False, suffix=".sh") as temp_script:
        temp_script.write(script_content)
        temp_script_path = temp_script.name

    # Make the temporary script executable
    os.chmod(temp_script_path, 0o755)

    # Construct the command
    command = [temp_script_path]

    if debug_mode:
        # Print the command instead of running it
        print("Debug Mode: The following command would be executed:")
        print(" ".join(command))
        print("Script content:")
        print(script_content)
    else:
        # Run the temporary shell script
        result = subprocess.run(command, capture_output=True, text=True)

        # Print the output and error messages
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

    # Clean up the temporary script
    os.remove(temp_script_path)

    print("Enhanced file saved at:", output_file)
    return output_file


def main():
    python_path = "/mn/stornext/d9/data/avijeetp/envs/enhance/bin/python"
    script_dir = "/mn/stornext/u3/avijeetp/codes/enhance"
    input_dir = "/mn/stornext/u3/avijeetp/codes/pyMilne/example_CRISP/temp"
    output_dir = input_dir
    input_filename = "hmi.ic_45s.20200807_082230_TAI.2.continuum.fits"

    input_file = os.path.join(input_dir, input_filename)
    output_file = os.path.join(output_dir, "enhanced_" + input_filename)

    file_type = "intensity"
    debug_mode = False

    run_enhance(python_path, script_dir, input_file, file_type, output_file, debug_mode)


if __name__ == "__main__":
    main()
