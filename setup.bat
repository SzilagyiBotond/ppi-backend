@echo off
REM Define the Conda environment name
SET ENV_NAME=ppi_backend

REM init conda if it was not done before
CALL conda init

REM Create a new Conda environment with Python
conda create -y -n %ENV_NAME% python=3.10 pip

REM Deactivate first, because sometimes it bugs (for me, atleast)
CALL conda deactivate

REM Activate the environment
CALL conda activate %ENV_NAME%

REM Confirm success
echo Conda environment "%ENV_NAME%" set up successfully with dependencies from poetry.lock.
