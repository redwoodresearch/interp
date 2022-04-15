#!/bin/sh

npm run rollup
PYTHONPATH=../../ python upload.py

