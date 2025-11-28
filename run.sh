#!/bin/bash


PREFIX="docker run --user=worker --gpus all --env DISPLAY=$DISPLAY --env="MPLCONFIGDIR=/tmp/matplotlib" --env="NVIDIA_DRIVER_CAPABILITIES=all" --shm-size=10.24gb --volume /tmp/.X11-unix:/tmp/.X11-unix:rw --volume $(pwd):/home/worker/work --volume $(pwd)/../datasets:/home/worker/work/datasets --rm --interactive --tty meetingdocker/ml:ripple2vec python -X pycache_prefix=./cache -m src.main --num-walks 5 --walk-length 15 --window-size 10 --dimensions 32 --OPT1 True --OPT2 True --OPT3 True --OPT4 True --until-layer 4"



$PREFIX --input graph/brazil-airports.edgelist --output emb/brazil-ripple.emb