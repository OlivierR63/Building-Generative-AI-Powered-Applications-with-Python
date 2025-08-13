#!/bin/bash
pip3 install virtualenv

# create a virtual environment my_env
virtualenv my_env

# activate my_env
source my_env/bin/activate

# installing required libraries in my_env
pip install langchain==0.1.11 gradio==5.23.2 transformers==4.38.2 bs4==0.0.2 requests==2.31.0 torch==2.2.1