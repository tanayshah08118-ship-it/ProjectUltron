How to set up your environment

To keep your project clean and avoid modifying global system packages, create a virtual environment inside your project folder (c:\ProjectUltron) and install NumPy:

Open PowerShell or Command Prompt in c:\ProjectUltron.

1}Create a virtual environment:
In powershell -->
py -3.14 -m venv .venv

2}Activate the virtual environment:
In powershell -->
.\.venv\Scripts\Activate.ps1

(If PowerShell gives an execution policy warning, you can run Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass first, or activate via Command Prompt with .\.venv\Scripts\activate.bat)

3}Install NumPy inside the environment:
In powershell -->
pip install numpy

4}Verify the installation:
In powershell -->
python -c "import numpy; print('NumPy successfully loaded:', numpy.__version__)"
