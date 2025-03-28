#!/usr/bin/env python
# filepath: b:\model\julia\fourier\Almass\machine_learning\ml_project\mlfit.py
import sys
import os

# Ensure the script can import from the project directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    main()