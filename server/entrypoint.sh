#!/bin/sh
bert-serving-start -http_port 8125 -num_worker=$1 -model_dir /model/cased_L-24_H-1024_A-16