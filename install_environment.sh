#!/usr/bin/env bash
set -f
rm -rf $(pipenv --venv)
pipenv run pip install "pip==18.1"
pipenv sync
pipenv run pip3 install --force-reinstall "torch==0.4.1"