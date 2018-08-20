#!/bin/sh

# ARS tuning
echo "ARS tuning"
python -m ars.tune_lqr_ars --H_start=1 --H_end=10 --H_bin=1
echo "ARS one direction tuning"
python -m ars.tune_lqr_ars --H_start=1 --H_end=10 --H_bin=1 --use_one_direction

# Exact tuning
echo "Exact one direction tuning"
python -m exact.tune_lqr_exact --H_start=1 --H_end=10 --H_bin=1 --use_one_direction
echo "Exact tuning"
python -m exact.tune_lqr_exact --H_start=1 --H_end=10 --H_bin=1

# ARS run
echo "ARS run"
python -m ars.run_ars_lqr_script --H_start=1 --H_end=10 --H_bin=1
echo "ARS one direction run"
python -m ars.run_ars_lqr_script --H_start=1 --H_end=10 --H_bin=1 --use_one_direction

# Exact run
echo "Exact run"
python -m exact.run_exact_lqr_script --H_start=1 --H_end=10 --H_bin=1
echo "Exact one direction run"
python -m exact.run_exact_lqr_script --H_start=1 --H_end=10 --H_bin=1 --use_one_direction

# Reinforce run
echo "Reinforce run"
python -m reinforce.run_lqr_reinforce_script --H_start=1 --H_end=10 --H_bin=1

echo "DONE"
