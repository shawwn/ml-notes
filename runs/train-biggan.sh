#!/bin/bash

set -ex

#exec bash repl0.sh main_biggan.py --gin_config configs/biggan_run01.gin --params 117M.json
#exec bash repl0.sh train_biggan.py --gin_config configs/biggan_run01.gin
#exec bash replpen.sh train_biggan.py --gin_config configs/biggan_run01.gin
exec bash repl0.sh train_biggan.py --gin_config configs/biggan_run01.gin

