import world
from world import SamplingAlgorithms
import torch
from torch.utils.data import DataLoader
import model
import utils
import dataloader
import numpy as np
import TrainProcedure
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint
import os
import time

# loading data...
dataset   = dataloader.LastFM()
lm_loader = DataLoader(dataset, batch_size=world.config['batch_size'], shuffle=True, drop_last=True) 

world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items

print('===========config================')
pprint(world.config)
print(world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print(world.sampling_type)
print('===========end===================')

# initialize models


if world.sampling_type == SamplingAlgorithms.alldata:
    print('all data train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN_xij(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel,
                      var_model=Varmodel)

elif world.sampling_type == SamplingAlgorithms.Alldata_train_ELBO:
    print('Alldata_train_ELBO train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_xij_sig(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel,
                      var_model=Varmodel)

elif world.sampling_type == SamplingAlgorithms.uniform:
    Recmodel = model.RecMF(world.config)
    elbo     = utils.BCE(Recmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-uniform.pth.tar')))
        Recmodel.train()

elif world.sampling_type == SamplingAlgorithms.sampler:
    Recmodel = model.RecMF(world.config)
    # Varmodel = model.VarMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    # register ELBO loss
    elbo = utils.ELBO(world.config, 
                    rec_model=Recmodel, 
                    var_model=Varmodel)
    sampler = utils.Sample_MF(k=1, var_model=Varmodel) 
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-sampler.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-sampler.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.bpr:
    Recmodel = model.RecMF(world.config)
    bpr = utils.BPRLoss(Recmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-bpr.pth.tar')))



elif world.sampling_type == SamplingAlgorithms.GMF:
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler = utils.sample_for_basic_GMF_loss(k=1.5)

elif world.sampling_type == SamplingAlgorithms.light_gcn:
    print('sampling_LGN')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler = utils.sample_for_basic_GMF_loss(k=3)

elif world.sampling_type == SamplingAlgorithms.Mixture:
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler_GMF = utils.sample_for_basic_GMF_loss(k=1.5)
    sampler_fast = utils.Sample_MF(k=1, var_model=Varmodel) # k doesn't matter
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-Mixture.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-Mixture.pth.tar')))
elif world.sampling_type == SamplingAlgorithms.Sample_positive_all:
    print('Sample_positive_all')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel, var_model=Varmodel)
    sampler = utils.Sample_positive_all(dataset, Varmodel)

else:
    print('sampling_LGN_mixture')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler1 = utils.sample_for_basic_GMF_loss(k=3)
    sampler2 = utils.Sample_MF(k=1, var_model=Varmodel)


# train
Neg_k = 3
world.config['total_batch'] = int(len(dataset)/world.config['batch_size'])


if world.tensorboard:
    w : SummaryWriter = SummaryWriter("./output/"+ "runs/"+time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
try:
    bar = tqdm(range(world.TRAIN_epochs))
    for i in bar:
        # for batch_i, batch_data in tqdm(enumerate(lm_loader)):
        if world.sampling_type == SamplingAlgorithms.alldata:
            output_information = TrainProcedure.Alldata_train_set_gamma_cross_entrophy(dataset, Recmodel, elbo, i, w=w)

        elif world.sampling_type == SamplingAlgorithms.Alldata_train_ELBO:
            output_information = TrainProcedure.Alldata_train_ELBO(dataset, Recmodel, Varmodel, elbo, i, w=w)

        elif world.sampling_type == SamplingAlgorithms.uniform:
            # bar.set_description('[training]')
            output_information = TrainProcedure.uniform_train(dataset, lm_loader, Recmodel, elbo, Neg_k, i, w)
        elif world.sampling_type == SamplingAlgorithms.Sample_positive_all:

            epoch_k = dataset.trainDataSize * 4

            output_information = TrainProcedure.Sample_positive_all_LGN(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i, w)
            print('end one epoch spa')
        elif world.sampling_type == SamplingAlgorithms.sampler:
            epoch_k = dataset.n_users*5
            # epoch_k = dataset.trainDataSize*5
            # bar.set_description(f"[Sample {epoch_k}]")
            output_information = TrainProcedure.sampler_train_no_batch(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i, w)
        elif world.sampling_type == SamplingAlgorithms.bpr:
            output_information = TrainProcedure.BPR_train(dataset, lm_loader, Recmodel, bpr, i, w)



        elif world.sampling_type == SamplingAlgorithms.GMF:
            output_information = TrainProcedure.sampler_train_GMF(dataset, sampler, Recmodel, Varmodel, elbo, i, w)
        elif world.sampling_type == SamplingAlgorithms.light_gcn:
            #epoch_k = dataset.n_users*5

            epoch_k = dataset.trainDataSize*4
            output_information = TrainProcedure.sampler_train_LGN(dataset, sampler, Recmodel, Varmodel, elbo, i, w)

            print('over train and follow bar')
        elif world.sampling_type == SamplingAlgorithms.Mixture:
            output_information = TrainProcedure.sampler_train_Mixture_GMF(dataset, sampler_GMF, sampler_fast, Recmodel, Varmodel, elbo, i, w)
        else:
            epoch_k = 166972
            output_information = TrainProcedure.sampler_train_no_batch_LGN_mixture(dataset, sampler1, sampler2, Recmodel, Varmodel, elbo, epoch_k, i, w)
            #if i == 150:
                #params = list(Varmodel.named_parameters())  # get the index by debuging
                #print(params[0][0])
                #np.savetxt('trained_weight.txt', params[0][1].data)



        bar.set_description(output_information)
        #torch.save(Recmodel.state_dict(), f"../checkpoints/Rec-{world.sampling_type.name}.pth.tar")
        #if globals().get('Varmodel'):
            #torch.save(Varmodel.state_dict(), f"../checkpoints/Var-{world.sampling_type.name}.pth.tar")
        if i%3 == 0 and i != 0:
            # test
            bar.set_description("[TEST]")
            testDict = dataset.getTestDict()
            TrainProcedure.Test(dataset, Recmodel, world.top_k, i, w)
finally:
    if world.tensorboard:
        w.close()
