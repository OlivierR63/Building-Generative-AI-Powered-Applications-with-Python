#!/bin/bash
source ../setup.bash


python3.11 -m pip install flask
python3.11 -m pip install flask_cors

python3.11 -m pip install transformers==4.38.2 torch==2.2.1