#!/bin/bash

#SBATCH --account=pi-lhansen
#SBATCH --job-name=runtd
#SBATCH --output=./job-outs/WZVtilde/chiUnderline_0.5_a_e_0.15_a_h_0.05_gamma_e_1.0_gamma_h_3.0_psi_e_0.5_psi_h_0.5/XiE_layers_5_XiH_layers_5_kappa_layers_5_weight1_0.0_boundary1_2_weight2_0.0_boundary2_5_points_size_2_iter_num_15/runtd.out
#SBATCH --error=./job-outs/WZVtilde/chiUnderline_0.5_a_e_0.15_a_h_0.05_gamma_e_1.0_gamma_h_3.0_psi_e_0.5_psi_h_0.5/XiE_layers_5_XiH_layers_5_kappa_layers_5_weight1_0.0_boundary1_2_weight2_0.0_boundary2_5_points_size_2_iter_num_15/runtd.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=caslake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=56G

module load python/anaconda-2021.05

python3 NN_structure.py    --XiE_layers 5 --XiH_layers 5 --kappa_layers 5
python3 WZVtilde_BFGS.py   --chiUnderline 0.5 --a_e 0.15 --a_h 0.05 --gamma_e 1.0 --gamma_h 3.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.3 --XiE_layers 5 --XiH_layers 5 --kappa_layers 5 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 15 
python3 WZVtilde_variable.py   --chiUnderline 0.5 --a_e 0.15 --a_h 0.05 --gamma_e 1.0 --gamma_h 3.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.3 --XiE_layers 5 --XiH_layers 5 --kappa_layers 5 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 15 
python3 WZVtilde_moments.py   --chiUnderline 0.5 --a_e 0.15 --a_h 0.05 --gamma_e 1.0 --gamma_h 3.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.3 --XiE_layers 5 --XiH_layers 5 --kappa_layers 5 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 15 --W_fix 49 --Z_fix 14 --V_fix 0 --Vtilde_fix 14

