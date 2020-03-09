"""
Store global parameters here
"""
import torch
import os
from enum import Enum

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SamplingAlgorithms(Enum):
  uniform   = 1 # it sucks
  sampler   = 2
  bpr       = 3
  alldata   = 4
  GMF       = 5
  Mixture   = 6
  light_gcn = 7
  light_gcn_mixture =8
  Sample_positive_all = 9
  Alldata_train_ELBO = 10
sampling_type = SamplingAlgorithms.Alldata_train_ELBO

# hyperparameters 
config = {}
config['alpha'] = 100
config['beta']  = 20
config['eta']   = 0.5
config['epsilon'] = 0.001
config['latent_dim_rec'] = 16
config['latent_dim_var'] = 32
config['batch_size'] = 32768
config['bpr_batch_size'] = 4096
config['all_batch_size'] = 32768
config['lightGCN_n_layers']=2
config['xij_dim'] = 8
config['num_xij'] = 1

# ======================
TRAIN_epochs = 1000
LOAD = False
PATH = '../checkpoints'
top_k = 5
comment = f"MF_{sampling_type.name}"
tensorboard = True
GPU = torch.cuda.is_available()


# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)





# parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rlr",
                    type=float,
                    default=0.003,
                    help="recmodel learning rate")
parser.add_argument("--vlr", 
                    type=float,
                    default=0.001,
                    help="varmodel learning rate")
parser.add_argument('--batch',
                    type=int, 
                    default=64)




logo = r"""
███████╗ █████╗ ███╗   ███╗██████╗ ██╗     ██╗███╗   ██╗ ██████╗ 
██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██║████╗  ██║██╔════╝ 
███████╗███████║██╔████╔██║██████╔╝██║     ██║██╔██╗ ██║██║  ███╗
╚════██║██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██║██║╚██╗██║██║   ██║
███████║██║  ██║██║ ╚═╝ ██║██║     ███████╗██║██║ ╚████║╚██████╔╝
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
print(logo)