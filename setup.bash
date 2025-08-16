#!/bin/bash

pip3 install virtualenv

# create a virtual environment my_env
virtualenv my_env

 # activate my_env
source my_env/bin/activate

python3 -m pip install transformers torch