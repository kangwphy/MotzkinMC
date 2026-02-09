#!/bin/bash
#SBATCH -p long
#SBATCH --job-name=S1
##SBATCH --output=logs/N%j.out
##SBATCH --error=logs/N%j.err
##SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --mem=100gb		#占用节点全部内存

module purge
module load compiler/devtoolset/7.3.1
module load mpi/hpcx/2.11.0/gcc-7.3.1
# Define the number of processes
NUM_PROCS=30

# # Check if command-line arguments are provided
# if [ "$#" -lt 3 ]; then
#     echo "Usage: ./run.sh L T Tthermal"
#     echo "Example: ./run.sh 10000 100000 10000"
#     exit 1
# fi

L=$3
# T=500000000
# Tthermal=1000000000
T=$4
Tthermal=$5
# T=20000
# Tthermal=10000
taskid=$1

# export JULIA_DEPOT_PATH="~/.julia_old"
# srun julia main.jl 
idx=$6
S=$2
echo $taskid $S $L $T $Tthermal
file=main.jl
task=basic
# 把错误输出到同一个文件 (Redirecting error to the same file)
# Run the Julia script with mpiexec, redirecting both stdout and stderr
mpiexec -n $NUM_PROCS julia code/$file $L $T $Tthermal $taskid $S $idx $task > logs/L${L}S${S}T${T}Tthm${Tthermal}_id${taskid}_idx${idx}_${task} 2>&1  &
wait