import os
import sys

# Dynamically find the library path (relative to the wrapper script)
_lib_path = os.path.join(os.path.dirname(__file__), "../lib")
sys.path.append(_lib_path)

from utils_lidar import *  # Import the C++ extension
