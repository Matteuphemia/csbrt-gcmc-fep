#!/usr/bin/env bash
set -u

echo "Host and working directory"
hostname
pwd

echo "Schedulers and environment managers"
command -v sbatch || true
command -v srun || true
command -v mamba || true
command -v micromamba || true
command -v conda || true

echo "GPU and driver"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi is unavailable on this node"
fi

echo "Conda or Mamba environments"
if command -v mamba >/dev/null 2>&1; then
  mamba env list
elif command -v micromamba >/dev/null 2>&1; then
  micromamba env list
elif command -v conda >/dev/null 2>&1; then
  conda env list
else
  echo "No Conda/Mamba executable is currently on PATH"
fi

echo "Slurm partitions"
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o '%P %G %l %a'
else
  echo "sinfo is unavailable on this node"
fi
