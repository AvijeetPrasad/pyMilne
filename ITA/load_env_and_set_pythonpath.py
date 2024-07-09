# load_env_and_set_path.py
from dotenv import load_dotenv, find_dotenv
import os
import sys


def setup_pythonpath_from_env():
    """
    Load PYTHONPATH from .env file and update sys.path accordingly.
    If PYTHONPATH is already set in the environment, it will be overridden by the value from .env.
    """
    # Find and load the .env file
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)

    # Check if PYTHONPATH is already set
    existing_pythonpath = os.getenv('PYTHONPATH')

    # Ensure PYTHONPATH is set from .env if it exists
    if dotenv_path:
        load_dotenv(dotenv_path, override=True)
        pythonpath = os.getenv('PYTHONPATH')
    else:
        pythonpath = existing_pythonpath

    # Split PYTHONPATH into individual paths and add to sys.path
    if pythonpath:
        for path in pythonpath.split(':'):
            if path not in sys.path:
                sys.path.append(path)

    return pythonpath


def print_pythonpath(verbose=False):
    """
    Print the current PYTHONPATH.
    """
    pythonpath = os.getenv('PYTHONPATH')
    print("Using PYTHONPATH from .env:", pythonpath)
    if verbose:
        # Print sys.path to verify the paths are added
        print("Updated sys.path:", sys.path)


def safe_import(module_name, attribute_name=None):
    """
    Safely import a module or an attribute from a module and handle ImportError if the module is not found.

    Parameters:
    - module_name (str): The name of the module to import.
    - attribute_name (str or None): The name of the attribute to import from the module.

    Returns:
    - module or attribute: The imported module or attribute if successful, None otherwise.
    """
    try:
        if attribute_name:
            module = __import__(module_name, fromlist=[attribute_name])
            print(f"Executing: from {module_name} import {attribute_name}")
            return getattr(module, attribute_name)
        else:
            print(f"Executing: import {module_name}")
            return __import__(module_name)
    except ImportError as error:
        print("")
        print("[!] " + error.__class__.__name__ + ": " + error.msg)
        print("")
        print(f"[!] Looks like you don't have the {module_name} package")
        print("")
        return None


# Automatically run the setup function on import
pythonpath = setup_pythonpath_from_env()
print_pythonpath()

# Explicitly export functions
__all__ = ['setup_pythonpath_from_env', 'print_pythonpath', 'safe_import']
