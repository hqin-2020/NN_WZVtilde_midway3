#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=runtd
#SBATCH --output=./job-outs/WZVtilde/chiUnderline_1.0_a_e_0.14_a_h_0.135_gamma_e_1.0_gamma_h_1.0_psi_e_0.5_psi_h_0.5/XiE_layers_5_XiH_layers_5_kappa_layers_5_weight1_0.0_boundary1_2_weight2_0.0_boundary2_5_points_size_10_iter_num_10/runtd.out
#SBATCH --error=./job-outs/WZVtilde/chiUnderline_1.0_a_e_0.14_a_h_0.135_gamma_e_1.0_gamma_h_1.0_psi_e_0.5_psi_h_0.5/XiE_layers_5_XiH_layers_5_kappa_layers_5_weight1_0.0_boundary1_2_weight2_0.0_boundary2_5_points_size_10_iter_num_10/runtd.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G

module load python/anaconda-2021.05
module load cuda/11.5

python3 NN_structure.py    --XiE_layers 5 --XiH_layers 5 --kappa_layers 5
python3 WZVtilde_BFGS.py   --chiUnderline 1.0 --a_e 0.14 --a_h 0.135 --gamma_e 1.0 --gamma_h 1.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.3 --XiE_layers 5 --XiH_layers 5 --kappa_layers 5 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 10 --iter_num 10 
python3 WZVtilde_variable.py   --chiUnderline 1.0 --a_e 0.14 --a_h 0.135 --gamma_e 1.0 --gamma_h 1.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.3 --XiE_layers 5 --XiH_layers 5 --kappa_layers 5 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 10 --iter_num 10 
python3 WZVtilde_moments.py   --chiUnderline 1.0 --a_e 0.14 --a_h 0.135 --gamma_e 1.0 --gamma_h 1.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.3 --XiE_layers 5 --XiH_layers 5 --kappa_layers 5 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 10 --iter_num 10 --W_fix 49 --Z_fix 14 --V_fix 0 --Vtilde_fix 14 

