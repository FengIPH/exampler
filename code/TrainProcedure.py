"""
Design training process
"""
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
          
          
FAST_SAMPLE_START = 100
          
          
def Alldata_train_set_gamma_cross_entrophy(dataset, recommend_model, loss_class, epoch, w=None):
    print('begin Alldata_train_set_gamma_cross_entrophy!')
    Recmodel : model.RecMF = recommend_model
    loss_class : utils.ELBO
    Recmodel.train()
    gamma = torch.ones((dataset.n_users, dataset.m_items))*0.5
    (epoch_users,
     epoch_items,
     epoch_xij,
     epoch_gamma) = utils.getAllData(dataset, gamma)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)
    rating = Recmodel(epoch_users, epoch_items)
    print(epoch_users[:1000], epoch_items[:1000], epoch_xij[:1000], rating[:1000])
    loss1 = loss_class.stageOne(rating, epoch_xij, epoch_gamma)

    # for (batch_i, (batch_users, batch_items, batch_xij, batch_gamma)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij, epoch_gamma)):
    # if epoch == 0:
    # print(len(batch_users))

    # rating = Recmodel(batch_users, batch_items)
    # loss1 = loss_class.stageOne(rating, batch_xij, batch_gamma)

    # if batch_i == 99:
        #print(batch_users[:1000], batch_items[:1000], batch_xij[:1000], rating[:1000], batch_gamma[:1000])

    # if world.tensorboard:
        # w.add_scalar(f'Alldata_train_set_gamma_cross_entrophy/stageOne', loss1, epoch*world.config['total_batch'] + batch_i)
    if world.tensorboard:
        w.add_scalar("Alldata_train_set_gamma_cross_entrophy/stageOne", loss1, epoch)

    print('end Alldata_train_set_gamma_cross_entrophy!')
    return f"[ALL[{datalen}]]"


    
def BPR_train(dataset, loader,recommend_model, loss_class, epoch, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr : utils.BPRLoss = loss_class  
    allusers = list(range(dataset.n_users))        
    S, sam_time = utils.UniformSample_allpos(allusers, dataset)
    users = torch.Tensor(S[:,0]).long()
    posItems = torch.Tensor(S[:,1]).long()
    negItems = torch.Tensor(S[:,2]).long()
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    for (batch_i, 
        (batch_users, 
         batch_pos, 
         batch_neg)) in enumerate(utils.minibatch(users, 
                                                  posItems, 
                                                  negItems, 
                                                  batch_size=world.config['bpr_batch_size'])):
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch*int(len(users)/world.config['bpr_batch_size']) + batch_i)

    # for batch_i, batch_data in enumerate(loader):
    #     if batch_i == 0:
    #         print(batch_data[:5])
    #     users = batch_data.numpy() # (batch_size, 1)
    #     # S, sam_time = utils.UniformSample(users, dataset, k=1)
    #     S, sam_time = utils.UniformSample_allpos(users, dataset)
        
    #     assert S.shape[-1] == 3
    #     print("users => S:", len(users),len(S))
    #     posItems = S[:, 1]
    #     negItems = S[:, 2]
        
    #     users = torch.Tensor(S[:,0]).long()
    #     posItems = torch.Tensor(posItems).long()
    #     negItems = torch.Tensor(negItems).long()
    #     cri = bpr.stageOne(users, posItems, negItems)
        
    #     if world.tensorboard:
    #         w.add_scalar(f'BPRLoss/BPR', cri, epoch*world.config['total_batch'] + batch_i)
    return f"[BPR[{cri:.3e}]]"

def Alldata_train_ELBO(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    print('begin Alldata_train_ELBO!')
    Recmodel: model.RecMF = recommend_model
    Varmodel: model.VarMF_xij_sig = var_model
    loss_class: utils.ELBO


    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)
    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):

        if epoch == 0:
            print(len(batch_users))
        Recmodel.train()
        Varmodel.eval()
        rating = Recmodel(batch_users, batch_items)
        batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss1 = loss_class.stageOne(rating, batch_xij, batch_gamma)

        Recmodel.eval()
        Varmodel.train()
        rating = Recmodel(batch_users, batch_items)
        batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        if batch_i == 24:
            print(batch_xij[:1000], rating[:1000], batch_gamma[:1000])
            print(batch_xij[-1000:], rating[-1000:], batch_gamma[-1000:])
        loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij)

        if world.tensorboard:
            w.add_scalar(f'Alldata_train_ELBO/stageOne', loss1, epoch*world.config['total_batch'] + batch_i)
            w.add_scalar(f'Alldata_train_ELBO/stageTwo', loss2, epoch*world.config['total_batch'] + batch_i)

    print('end Alldata_train_ELBO!')
    return f"[ALL[{datalen}]]"
    

def uniform_train(dataset, loader,recommend_model, loss_class, Neg_k, epoch, w=None):
    # batch_data = user_batch
    Recmodel = recommend_model
    Recmodel.train()
    bce : utils.BCE = loss_class
    total_start = time()
    sampling_time = 0.
    process_time = 0.
    train_time = 0.
    
    for batch_i, batch_data in enumerate(loader):
        
        users = batch_data.numpy() # (batch_size, 1)
        # 1.
        # sample items
        start = time()
        S, sam_time = utils.UniformSample(users, dataset, k=Neg_k)
        # print(sam_time)
        end  = time()
        sampling_time += (end-start)
        
        start = time()
        users = np.tile(users.reshape((-1,1)), (1, 1+Neg_k)).reshape(-1)
        S     = S.reshape(-1)
        # 2.
        # process xij
        xij   = dataset.getUserItemFeedback(users, S)
        
        users = torch.Tensor(users).long()
        items = torch.Tensor(S).long()
        xij   = torch.Tensor(xij)
        end = time()
        process_time += (end-start)       
        # 3.
        # optimize loss
        # start = time()
        start = time()
        rating = Recmodel(users, items)
        loss1  = bce.stageOne(rating, xij)
        end = time()
        train_time += (end-start)
        # end   = time()
        if world.tensorboard:
            w.add_scalar(f'UniformLoss/BCE', loss1, epoch*world.config['total_batch'] + batch_i)
    total_end = time()
    total_time = total_end - total_start
    return f"[UNI]{total_time:.1f}={sampling_time:.1f}+{process_time:.1f}+{train_time:.1f}"


# users_set = set()
# items_set = set()
def sampler_train_no_batch(dataset, sampler, recommend_model, var_model_reg, loss_class, epoch_k, epoch,w):
    # global users_set, items_set
    sampler : utils.Sample_MF
    dataset : dataloader.BasicDataset
    recommend_model.train()
    var_model_reg.train()
    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    sampler.compute()
    epoch_users, epoch_items = sampler.sampleForEpoch(epoch_k) # epoch_k may be 5*n

    epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                            epoch_items.cpu().numpy()).astype('int')
    # print(f"[{epoch}]Positive Label Sparsity",np.sum(epoch_xij)/len(epoch_xij))
    # print(epoch_users[:5], epoch_items[:5], epoch_xij[:5])
    epoch_users = epoch_users.long()
    epoch_items = epoch_items.long()
    epoch_xij   = torch.Tensor(epoch_xij)
    
    epoch_rating = recommend_model(epoch_users, epoch_items)
    loss1 = loss_class.stageOne(epoch_rating, epoch_xij)
    
    epoch_rating = recommend_model(epoch_users, epoch_items)
    epoch_gamma  = var_model_reg(epoch_users, epoch_items)
    
    loss2  = loss_class.stageTwo(epoch_rating, epoch_gamma, epoch_xij)

    if world.tensorboard:
        w.add_scalar(f'SamplerLoss/stageOne', loss1, epoch)
        w.add_scalar(f'SamplerLoss/stageTwo', loss2, epoch)
    return f"Sparsity {(torch.sum(epoch_xij)/len(epoch_xij)).item():.3f}"


def sampler_train(dataset, sampler, recommend_model, var_model_reg, loss_class, epoch_k, epoch,w):
    # global users_set, items_set
    sampler : utils.Sample_MF
    dataset : dataloader.BasicDataset
    recommend_model.train()
    var_model_reg.train()
    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    sampler.compute()
    epoch_users, epoch_items = sampler.sampleForEpoch(epoch_k) # epoch_k may be 5*n

    epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                            epoch_items.cpu().numpy()).astype('int')
    # print(f"[{epoch}]Positive Label Sparsity",np.sum(epoch_xij)/len(epoch_xij))
    # print(epoch_users[:5], epoch_items[:5], epoch_xij[:5])
    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        users = batch_users.long()
        # print(users.size())
        items = batch_items.long()
        xij   = torch.Tensor(batch_xij)
        
        rating = recommend_model(users, items)
        loss1  = loss_class.stageOne(rating, xij)

        rating = recommend_model(users, items)
        gamma  = var_model_reg(users, items)

        loss2  = loss_class.stageTwo(rating, gamma, xij)
        # end = time()
        # print(f"{world.sampling_type } opt time", end-start)
        if world.tensorboard:
            w.add_scalar(f'SamplerLoss/stageOne', loss1, epoch*world.config['total_batch'] + batch_i)
            w.add_scalar(f'SamplerLoss/stageTwo', loss2, epoch*world.config['total_batch'] + batch_i)
    return f"Sparsity {np.sum(epoch_xij)/len(epoch_xij):.3f}"


def Test(dataset, Recmodel, top_k, epoch, w=None):
    dataset : utils.BasicDataset
    testDict : dict = dataset.getTestDict()
    Recmodel : model.RecMF
    with torch.no_grad():
        Recmodel.eval()
        users = torch.Tensor(list(testDict.keys()))
        GroundTrue = [testDict[user] for user in users.numpy()]
        rating = Recmodel.getUsersRating(users)
        # exclude positive train data
        allPos = dataset.getUserPosItems(users)
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i]*len(items))
            exclude_items.extend(items)
        rating[exclude_index, exclude_items] = 0.
        # assert torch.all(rating >= 0.)
        # assert torch.all(rating <= 1.)
        # end excluding
        _, top_items = torch.topk(rating, top_k)
        top_items = top_items.cpu().numpy()
        metrics = utils.recall_precisionATk(GroundTrue, top_items, top_k)
        metrics['mrr'] = utils.MRRatK(GroundTrue, top_items, top_k)
        metrics['ndcg'] = utils.NDCGatK(GroundTrue, top_items, top_k)
        print(metrics)
        if world.tensorboard:
            w.add_scalar(f'Test/Recall@{top_k}', metrics['recall'], epoch)
            w.add_scalar(f'Test/Precision@{top_k}', metrics['precision'], epoch)
            w.add_scalar(f'Test/MRR@{top_k}', metrics['mrr'], epoch)
            w.add_scalar(f'Test/NDCG@{top_k}', metrics['ndcg'], epoch)
            

def sampler_train_GMF(dataset, sampler, recommend_model, var_model_reg, loss_class, epoch, w):
    # global users_set, items_set
    sampler : utils.sample_for_basic_GMF_loss
    dataset : dataloader.BasicDataset
    recommend_model.train()
    var_model_reg.train()
    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    #sampler.compute()
    title = "GMF"
    epoch_users, epoch_items = sampler.sampleForEpoch(dataset, k=1.5)

    epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                                epoch_items.cpu().numpy()).astype('int')    

    for (batch_i, 
         (batch_users, 
          batch_items, 
          batch_xij)) in enumerate(utils.minibatch(epoch_users, 
                                                   epoch_items, 
                                                   epoch_xij)):
        users = batch_users.long()
        # print(users.size())
        items = batch_items.long()
        xij   = torch.Tensor(batch_xij)
        gamma = var_model_reg(users, items)
        rating = recommend_model(users, items)

        loss1  = loss_class.stageOne(rating, xij, gamma)

        rating = recommend_model(users, items)
        loss2  = loss_class.stageTwo(rating, gamma, xij)
        # end = time()
        # print(f"{world.sampling_type } opt time", end-start)
        if batch_i == world.config['total_batch']:
            print()
            print(f'{title:}')
            # print(batch_users[:10])
            # print(batch_items[:10])
            pprint(batch_xij[:6])
            pprint(rating[:6])
            pprint(gamma[:6])
        if world.tensorboard:
            w.add_scalar(f'SamplerLoss/stageOne', loss1, epoch*(int(len(epoch_users)/world.config['batch_size']) + 1) + batch_i)
            w.add_scalar(f'SamplerLoss/stageTwo', loss2, epoch*(int(len(epoch_users)/world.config['batch_size']) + 1) + batch_i)
    return f"[{title}]Sparsity{np.sum(epoch_xij)/len(epoch_xij):.3f}"
    
    
def sampler_train_Mixture_GMF(dataset, sampler_GMF, sampler_fast, recommend_model, var_model_reg, loss_class, epoch, w):
    # global users_set, items_set
    sampler_GMF : utils.sample_for_basic_GMF_loss
    sampler_fast : utils.Sample_MF
    dataset : dataloader.BasicDataset
    recommend_model.train()
    var_model_reg.train()
    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    #sampler.compute()
    title = "GMF"
    if epoch <= FAST_SAMPLE_START:
        epoch_users, epoch_items = sampler_GMF.sampleForEpoch(dataset, k=1.5)

        epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
        epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                                epoch_items.cpu().numpy()).astype('int')
        pos_items = np.sum(epoch_xij)
    else:
        # TODO
        epoch_k = dataset.n_users*10
        title = f"FAST{epoch_k}"
        sampler_fast.compute()
        epoch_users, epoch_items = sampler_fast.sampleForEpoch(epoch_k) # epoch_k may be 5*n

        epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
        epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                                epoch_items.cpu().numpy()).astype('int')
        epoch_xij = torch.from_numpy(epoch_xij).float()
        pos_items = torch.sum(epoch_xij).item()

    for (batch_i, 
         (batch_users, 
          batch_items, 
          batch_xij)) in enumerate(utils.minibatch(epoch_users, 
                                                   epoch_items, 
                                                   epoch_xij)):
        users = batch_users.long()
        # print(users.size())
        items = batch_items.long()
        xij   = torch.Tensor(batch_xij)
        gamma = var_model_reg(users, items)
        rating = recommend_model(users, items)

        loss1  = loss_class.stageOne(rating, xij, gamma)

        rating = recommend_model(users, items)
        loss2  = loss_class.stageTwo(rating, gamma, xij)
        # end = time()
        # print(f"{world.sampling_type } opt time", end-start)
        if batch_i == world.config['total_batch']:
            print()
            print(f'{title:}')
            # print(batch_users[:10])
            # print(batch_items[:10])
            pprint(batch_xij[:6])
            pprint(rating[:6])
            pprint(gamma[:6])
        if world.tensorboard:
            w.add_scalar(f'SamplerLoss/stageOne', loss1, epoch*(int(len(epoch_users)/world.config['batch_size']) + 1) + batch_i)
            w.add_scalar(f'SamplerLoss/stageTwo', loss2, epoch*(int(len(epoch_users)/world.config['batch_size']) + 1) + batch_i)
    return f"[{title}]Sparsity{pos_items/len(epoch_xij):.3f}"

def sampler_train_LGN(dataset, sampler, recommend_model, var_model, loss_class, epoch, w):
    # global users_set, items_set
    sampler : utils.sample_for_basic_GMF_loss
    dataset : dataloader.BasicDataset

    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    #sampler.compute()
    print('sampler1')
    epoch_users, epoch_items = sampler.sampleForEpoch(dataset, k=3)  # epoch_k may be 5*n

    epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                            epoch_items.cpu().numpy()).astype('int')

    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(
            utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        users = batch_users.long()
        # print(users.size())
        items = batch_items.long()
        xij = torch.Tensor(batch_xij)
        recommend_model.train()
        var_model.eval()
        #user_emb = var_model.getUsersEmbedding(users)
        #print('1:3_1', user_emb)

        gamma = var_model(users, items)
        rating = recommend_model(users, items)

        loss1 = loss_class.stageOne(rating, xij, gamma)
        recommend_model.eval()
        var_model.train()
        gamma = var_model(users, items)
        rating = recommend_model(users, items)
        loss2 = loss_class.stageTwo(rating, gamma, xij)
        #user_emb = var_model.getUsersEmbedding(users)
        if batch_i == world.config['total_batch']:
            print('the last batch:', batch_users[:1000], batch_items[:1000], batch_xij[:1000], rating[:1000],
                  gamma[:1000])

        #print('1:3_2', user_emb)

        if world.tensorboard:
            w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageOne', loss1, epoch)
            w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageTwo', loss2, epoch)
    print('over one epoch!!!!!')
    return f"Sparsity {(np.sum(epoch_xij) / len(epoch_xij)).item():.3f}"



def sampler_train_no_batch_LGN(dataset, sampler, recommend_model, var_model, loss_class, epoch_k, epoch, w):
    # global users_set, items_set
    sampler: utils.Sample_MF
    dataset: dataloader.BasicDataset
    loss_class: utils.ELBO
    # 1.
    # sampling
    # start = time()
    #sampler.compute()
    print('begin')
    epoch_users, epoch_items = sampler.sampleForEpoch(epoch_k)  # epoch_k may be 5*n

    epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                            epoch_items.cpu().numpy()).astype('int')
    # print(f"[{epoch}]Positive Label Sparsity",np.sum(epoch_xij)/len(epoch_xij))
    # print(epoch_users[:5], epoch_items[:5], epoch_xij[:5])
    epoch_users = epoch_users.long()
    epoch_items = epoch_items.long()
    epoch_xij = torch.Tensor(epoch_xij)

    recommend_model.train()
    var_model.eval()
    gamma = var_model(epoch_users, epoch_items)
    rating = recommend_model(epoch_users, epoch_items)

    loss1 = loss_class.stageOne(rating, epoch_xij, gamma, gamma)
    recommend_model.eval()
    var_model.train()
    gamma = var_model(epoch_users, epoch_items)
    rating = recommend_model(epoch_users, epoch_items)
    loss2 = loss_class.stageTwo(rating, gamma, epoch_xij, gamma)

    if world.tensorboard:
        w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageOne', loss1, epoch)
        w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageTwo', loss2, epoch)

    print('over one epoch!!!!!')

    return f"Sparsity {(torch.sum(epoch_xij) / len(epoch_xij)).item():.3f}"

def sampler_train_no_batch_LGN_mixture(dataset, sampler1, sampler2, recommend_model, var_model, loss_class, epoch_k, epoch, w):
    sampler1: utils.sample_for_basic_GMF_loss
    sampler2: utils.Sample_MF
    dataset: dataloader.BasicDataset
    loss_class: utils.ELBO
    # 1.
    # sampling
    # start = time()
    # sampler.compute()
    title = "GMF"
    if epoch <= FAST_SAMPLE_START:
        print('sampler1')
        epoch_users, epoch_items = sampler1.sampleForEpoch(dataset, k=3)  # epoch_k may be 5*n

        epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
        epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                                epoch_items.cpu().numpy()).astype('int')

        for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
            users = batch_users.long()
            # print(users.size())
            items = batch_items.long()
            xij = torch.Tensor(batch_xij)
            recommend_model.train()
            var_model.eval()

            gamma = var_model(users, items)
            rating = recommend_model(users, items)

            loss1 = loss_class.stageOne(rating, xij, gamma)
            recommend_model.eval()
            var_model.train()
            gamma = var_model(users, items)
            rating = recommend_model(users, items)
            loss2 = loss_class.stageTwo(rating, gamma, xij)
            if batch_i == world.config['total_batch']:
                print('the last batch:', batch_users[:1000], batch_items[:1000], batch_xij[:1000], rating[:1000],
                      gamma[:1000])


            if world.tensorboard:
                w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageOne', loss1, epoch)
                w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageTwo', loss2, epoch)
        print('over one epoch!!!!!')
        return f"Sparsity {(np.sum(epoch_xij) / len(epoch_xij)).item():.3f}"


    else:
        epoch_users, epoch_items = sampler2.sampleForEpoch(epoch_k)  # epoch_k may be 5*n

        epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
        epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                                epoch_items.cpu().numpy()).astype('int')
        # print(f"[{epoch}]Positive Label Sparsity",np.sum(epoch_xij)/len(epoch_xij))
        # print(epoch_users[:5], epoch_items[:5], epoch_xij[:5])
        epoch_users = epoch_users.long()
        epoch_items = epoch_items.long()
        epoch_xij = torch.Tensor(epoch_xij)

        recommend_model.train()
        var_model.eval()
        gamma = var_model(epoch_users, epoch_items)
        rating = recommend_model(epoch_users, epoch_items)


        loss1 = loss_class.stageOne(rating, epoch_xij, gamma, gamma)
        recommend_model.eval()
        var_model.train()
        gamma = var_model(epoch_users, epoch_items)

        rating = recommend_model(epoch_users, epoch_items)
        print(epoch_xij[:1000], rating[:1000], gamma[:1000])
        loss2 = loss_class.stageTwo(rating, gamma, epoch_xij, gamma)

        if world.tensorboard:
            w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageOne', loss1, epoch)
            w.add_scalar(f'sampler_train_no_batch_LGN_mixture/stageTwo', loss2, epoch)

        print('over one epoch!!!!!')
        return f"Sparsity {(torch.sum(epoch_xij) / len(epoch_xij)).item():.3f}"




            #if world.tensorboard:
                #w.add_scalar(f'SamplerLoss/stageOne', loss1, epoch)
                #w.add_scalar(f'SamplerLoss/stageTwo', loss2, epoch)

def Sample_positive_all_LGN(dataset, sampler, recommend_model, var_model, loss_class, epoch_k, epoch, w):
    # global users_set, items_set
    print('begin spa')
    sampler: utils.Sample_positive_all
    dataset: dataloader.BasicDataset
    loss_class: utils.ELBO
    # 1.
    # sampling
    # start = time()
    #sampler.compute()
    epoch_users, epoch_items = sampler.sampleForEpoch(epoch_k)  # epoch_k may be 5*n

    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                            epoch_items.cpu().numpy()).astype('int')
    # print(f"[{epoch}]Positive Label Sparsity",np.sum(epoch_xij)/len(epoch_xij))
    # print(epoch_users[:5], epoch_items[:5], epoch_xij[:5])
    epoch_users = epoch_users.long()
    epoch_items = epoch_items.long()
    epoch_xij = torch.Tensor(epoch_xij)

    recommend_model.train()
    var_model.eval()
    gamma = var_model(epoch_users, epoch_items)
    rating = recommend_model(epoch_users, epoch_items)
    pij = sampler.sample_prob(gamma)



    loss1 = loss_class.stageOne(rating, epoch_xij, gamma, pij)
    recommend_model.eval()
    var_model.train()
    gamma = var_model(epoch_users, epoch_items)
    rating = recommend_model(epoch_users, epoch_items)
    pij = sampler.sample_prob(gamma)

    print('po', epoch_users[:500], epoch_items[:500], epoch_xij[:500], rating, gamma[:500])
    print('neg', epoch_users[-500:], epoch_items[-500:], epoch_xij[-500:], rating, gamma[-500:])
    loss2 = loss_class.stageTwo(rating, gamma, epoch_xij, pij)

    if world.tensorboard:
        w.add_scalar(f'Sample_positive_all_LGN/stageOne', loss1, epoch)
        w.add_scalar(f'Sample_positive_all_LGN/stageTwo', loss2, epoch)

    print('over one epoch!!!!!')

    return f"Sparsity {(torch.sum(epoch_xij) / len(epoch_xij)).item():.3f}"

