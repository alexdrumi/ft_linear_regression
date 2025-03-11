#!/bin/bash

########################################
# 1. Get the location of this script
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
SRC_DIR="${SCRIPT_DIR}/src"

########################################
# 2. Check & Install Python if needed
########################################
check_and_install_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Python3 is not installed. Installing Python..."

        # Install Python based on OS
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            if ! command -v brew &> /dev/null; then
                echo "Homebrew is not installed. Installing Homebrew..."
                /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"
            fi
            brew install python
        else
            echo "Unsupported OS. Please install Python manually."
            exit 1
        fi
    else
        echo "Python3 is already installed."
    fi
}

########################################
# 3. Create & activate the virtual environment
########################################
setup_venv() {
    # Make sure Python is present
    check_and_install_python

    # Create the venv if it doesn't exist
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment at ${VENV_DIR}..."
        python3 -m venv "${VENV_DIR}"
    else
        echo "Virtual environment already exists at ${VENV_DIR}."
    fi

    # Activate the venv
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"

    # Upgrade pip and install from requirements
    echo "Upgrading pip..."
    pip install --upgrade pip

    if [ -f "${REQUIREMENTS_FILE}" ]; then
        echo "Installing packages from ${REQUIREMENTS_FILE}..."
        pip install --upgrade -r "${REQUIREMENTS_FILE}"
    else
        echo "No requirements.txt found at ${REQUIREMENTS_FILE}."
        echo "If you need packages, please add them to requirements.txt."
    fi
}

########################################
# 4. Main Execution
########################################

# 4a. Set up the venv
setup_venv

# 4b. [Optional] Run your training/prediction scripts
# Replace the lines below with the specific Python scripts you want to run
echo "Running your Python scripts..."

cd "${SRC_DIR}"
python train_linear_regression.py
python predict_linear_regression.py

echo "All tasks are complete!"
