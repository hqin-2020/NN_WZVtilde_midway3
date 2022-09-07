import json
import tensorflow as tf 
import pandas as pd
import time 
import os
import mfr.sdm as sdm
from WZVtilde_para import *
from WZVtilde_training import *
import argparse

## Parameter parser
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--nWealth",type=int,default=100)
parser.add_argument("--nZ",type=int,default=30)
parser.add_argument("--nV",type=int,default=30)
parser.add_argument("--nVtilde",type=int,default=0)
parser.add_argument("--V_bar",type=float,default=1.0)
parser.add_argument("--Vtilde_bar",type=float,default=0.0)
parser.add_argument("--sigma_K_norm",type=float,default=0.04)
parser.add_argument("--sigma_Z_norm",type=float,default=0.0141)
parser.add_argument("--sigma_V_norm",type=float,default=0.132)
parser.add_argument("--sigma_Vtilde_norm",type=float,default=0.0)

parser.add_argument("--chiUnderline",type=float,default=1.0)
parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--psi_e",type=float,default=1.0)
parser.add_argument("--psi_h",type=float,default=1.0)

parser.add_argument("--XiE_layers",type=int,default=5)
parser.add_argument("--XiH_layers",type=int,default=5)
parser.add_argument("--kappa_layers",type=int,default=5)
parser.add_argument("--weight1",type=float,default=30.0)
parser.add_argument("--boundary1",type=int,default=2)
parser.add_argument("--weight2",type=float,default=100.0)
parser.add_argument("--boundary2",type=int,default=5)
parser.add_argument("--points_size",type=int,default=2)
parser.add_argument("--iter_num",type=int,default=10)

parser.add_argument("--W_fix",type=int,default=5)
parser.add_argument("--Z_fix",type=int,default=5)
parser.add_argument("--V_fix",type=int,default=5)
parser.add_argument("--Vtilde_fix",type=int,default=5)
args = parser.parse_args()

## Domain parameters
nWealth           = args.nWealth
nZ                = args.nZ
nV                = args.nV
nVtilde           = args.nVtilde
V_bar             = args.V_bar
Vtilde_bar        = args.Vtilde_bar
sigma_K_norm      = args.sigma_K_norm
sigma_Z_norm      = args.sigma_Z_norm
sigma_V_norm      = args.sigma_V_norm
sigma_Vtilde_norm = args.sigma_Vtilde_norm
domain_list       = [nWealth, nZ, nV, nVtilde, V_bar, Vtilde_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, sigma_Vtilde_norm]
if sigma_Vtilde_norm == 0:
  domain_folder = 'WZV'
  nDims = 3
elif sigma_V_norm == 0:
  domain_folder = 'WZVtilde'
  nDims = 3
else:
  domain_folder = 'WZVVtilde'
  nDims = 4

## Model parameters
chiUnderline      = args.chiUnderline
a_e               = args.a_e
a_h               = args.a_h
gamma_e           = args.gamma_e
gamma_h           = args.gamma_h
psi_e             = args.psi_e
psi_h             = args.psi_h
parameter_list    = [chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h]
chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

## NN layer parameters
XiE_layers        = args.XiE_layers
XiH_layers        = args.XiH_layers
kappa_layers      = args.kappa_layers
weight1           = args.weight1
boundary1         = args.boundary1
weight2           = args.weight2
boundary2         = args.boundary2
points_size       = args.points_size
iter_num          = args.iter_num
layer_folder = 'XiE_layers_' + str(XiE_layers) +'_XiH_layers_' + str(XiH_layers) +'_kappa_layers_'+ str(kappa_layers) + '_weight1_' + str(int(weight1)) + '_boundary1_' + str(boundary1) + '_weight2_' + str(int(weight2)) + '_boundary2_' + str(boundary2)+ '_points_size_' + str(points_size) + '_iter_num_' + str(iter_num) 

## Working directory
workdir = os.path.dirname(os.getcwd())
srcdir = workdir + '/src/'
datadir = workdir + '/data/' + domain_folder + '/' + model_folder + '/'
outputdir = workdir + '/output/' + domain_folder + '/' + model_folder + '/' + layer_folder + '/'
docdir = workdir + '/doc/' + domain_folder + '/' + model_folder + '/'+ layer_folder + '/'
os.makedirs(datadir,exist_ok=True)
os.makedirs(docdir,exist_ok=True)
os.makedirs(outputdir,exist_ok=True)

## Generate parameter set
setModelParameters(parameter_list, domain_list, nDims)
with open(datadir + 'parameters_NN.json') as json_file:
    paramsFromFile = json.load(json_file)
params = setModelParametersFromFile(paramsFromFile, nDims)

batchSize = 2048 * points_size
dimension = 3
units = 16
activation = 'tanh'
kernel_initializer = 'glorot_normal'

## NN structure
tf.keras.backend.set_floatx("float64") ## Use float64 by default

logXiE_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation= None,  kernel_initializer='glorot_normal')])

logXiH_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation= None , kernel_initializer='glorot_normal')])

kappa_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation='sigmoid', kernel_initializer='glorot_normal')])

## Training
start = time.time()
targets = tf.zeros(shape=(batchSize,1), dtype=tf.float64)
for iter in range(iter_num):
  W = tf.random.uniform(shape = (batchSize,1), minval = params['wMin'], maxval = params['wMax'], dtype=tf.float64)
  Z = tf.random.uniform(shape = (batchSize,1), minval = params['zMin'], maxval = params['zMax'], dtype=tf.float64)
  V = tf.random.uniform(shape = (batchSize,1), minval = params['vMin'], maxval = params['vMax'], dtype=tf.float64)
  Vtilde = tf.random.uniform(shape = (batchSize,1), minval = params['VtildeMin'], maxval = params['VtildeMax'], dtype=tf.float64)
  print('Iteration', iter)
  training_step_BFGS(logXiH_NN, logXiE_NN, kappa_NN, W, Z, V, Vtilde, params, targets, weight1, boundary1, weight2, boundary2)
end = time.time()
training_time = '{:.4f}'.format((end - start)/60)
print('Elapsed time for training {:.4f} sec'.format(end - start))

## Save trained neural network approximations and respective model parameters
tf.saved_model.save(logXiH_NN, outputdir   + 'logXiH_NN')
tf.saved_model.save(logXiE_NN, outputdir   + 'logXiE_NN')
tf.saved_model.save(kappa_NN,  outputdir   + 'kappa_NN')

NN_info = {'XiE_layers': XiE_layers, 'XiH_layers': XiH_layers, 'kappa_layers': kappa_layers, 'weight1': weight1, 'boundary1': boundary1, 'weight2': weight2, 'boundary2': boundary2,  'points_size': points_size,\
          'dimension': dimension, 'units': units, 'activation': activation, 'kernel_initializer': kernel_initializer, 'iter_num': iter_num, 'training_time': training_time}

with open(outputdir + "/NN_info.json", "w") as f:
  json.dump(NN_info,f)