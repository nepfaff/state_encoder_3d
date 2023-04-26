#!/bin/bash

# Script for training/ evaluation in non-interactive sessions on supercloud
# Submit via `LLsub supercloud_run.sh -s 20 -g volta:1` and show status via `LLstat`
# Logs are written to `supercloud_run.sh.log-{job_id}`

source /etc/profile

module load anaconda/2022b
module load cuda/11.6

# Replace this with the desired script
bash scripts/train_srn_ae.py
