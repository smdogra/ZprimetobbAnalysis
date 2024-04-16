#!/bin/bash
python3 -m notebook --no-browser --port=${1} --ip 127.0.0.1 &>./access_jupyter.log &
