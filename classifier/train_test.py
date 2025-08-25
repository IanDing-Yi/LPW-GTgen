#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import sys
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
from skimage import io, transform
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


from hist_dataloader import tiny_Dataset, Rescale, ToTensor, ProtoDataset, Normalize
from pretrain_model import get_pretrain_model


# In[3]:


torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# In[4]:


def train(clf, optimizer, trainloader, criterion, disp):
    count = 0
    policy_losses = []
    value_losses = []
    episode_reward = []
    if(disp):
        print(device)
    for i, data in enumerate(trainloader, 0):
        count += 1
        if device is None:
            inputs = data[0].type(torch.FloatTensor)
            labels = data[1].type(torch.FloatTensor)
        else:
            inputs = data[0].type(torch.FloatTensor).to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
        
        value_pred = clf(inputs)

        # label smoothing
        # 0->1 --> 0.05->0.95
        epsilon = 0.1
        smoothed_labels = labels * (1 - epsilon) + 0.5 * epsilon

        value_loss = criterion(value_pred.float(), smoothed_labels).sum()
        
#         if(disp):
#             print(value_loss)
        
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()
        
        value_losses.append(float(value_loss.item()))

    return sum(value_losses)/len(value_losses)


# evaluation
def comp_test(stage, clf, testloader, criterion, disp):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    preds = np.empty(0)
    lbs = np.empty(0)
    loss = []
    if(disp):
        print(device)
    with torch.no_grad():
        for data in testloader:
            if device is None:
                inputs = data[0]
                labels = data[1]
            else:
                inputs = data[0].to(device)
                labels = data[1].to(device)

            outputs = clf(inputs)
            # label smoothing
            epsilon = 0.1
            smoothed_labels = labels * (1 - epsilon) + 0.5 * epsilon

            val_loss = criterion(outputs.float(), smoothed_labels).sum()
            loss.append(val_loss.item())
#             predicted = torch.round(torch.sigmoid(outputs))
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            pred_npy = predicted.detach().cpu().numpy()
            total += labels.size(0)
            labels = torch.argmax(torch.softmax(labels, dim=-1), dim=-1)
            lb_npy = labels.detach().cpu().numpy()
            correct += (pred_npy == lb_npy).sum().item()
            preds = np.hstack((preds, pred_npy.squeeze()))
            lbs = np.hstack((lbs, lb_npy.squeeze()))

    conmx = confusion_matrix(lbs, preds)
    if(disp):
        print(stage+' accuracy: %.6f %%' % (100 * correct / total))
#     tn, fp, fn, tp = conmx.ravel()
#     if (tp + fp) == 0:
#         prec = 0
#     else:
#         prec = tp / (tp + fp)
#     if (tp + fn) == 0:
#         recl = 0
#     else:
#         recl = tp / (tp + fn)
#     if (prec+recl) == 0:
#         f1 = 0
#     else:
#         f1 = (2*prec*recl) / (prec+recl)
#     if(disp):
#         print('Precision:', prec)
#         print('Recall:', recl)
#         print('F1:', f1)
    return (correct / total), conmx, sum(loss)/len(loss)

def run_train(model_name, train_csv, val_csv, root_folder, save_path, disp, nb_cls=6, batch_size=10, lr=0.0001, patience=5, min_delta=0, max_episodes=1000):
    start_time = time.time()

    test_csv = val_csv

    train_dataset = tiny_Dataset(csv_file=train_csv,
                                 root_dir=root_folder,
                                 transform=transforms.Compose([
                                     Rescale((224,224)),
                                     ToTensor(),
                                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ]))
    test_dataset = tiny_Dataset(csv_file=test_csv,
                                root_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    clf = get_pretrain_model(model_name, nb_cls)
    clf.to(device)

    # pre-defined loss weights based on preliminary experiments
    # 1/class_precision
    cls_weights = torch.tensor([1.46993504, 1.83937636, 1.63301425, 1.10534349, 1., 1.]).to(device)

    criterion = nn.BCEWithLogitsLoss(weight=cls_weights)
    optimizer_clf = optim.AdamW(clf.parameters(), lr=lr)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_clf, mode='min', factor=0.5, patience=3, verbose=disp, min_lr=5e-7
    )

    max_test_perf = 10000
    min_delta = min_delta
    patience = patience
    counter = 0

    MAX_EPISODES = max_episodes
    PRINT_EVERY = 1

    records = {'train': [],'valid': []}
    for episode in range(1, MAX_EPISODES+1):  # loop over the dataset multiple times
        if(disp):
            print('episode:', episode)
        critic_loss = train(clf, optimizer_clf, trainloader, criterion, disp)
        if(disp):
            print('Train')
        tr_cur_acc, tr_conmx, tr_loss = comp_test('Train', clf, trainloader, criterion, disp)
        records['train'].append([tr_cur_acc, tr_conmx, tr_loss])
        if(disp):
            print('train loss: ', critic_loss)
            print('train loss: ', tr_loss)
            print('Validation')
        cur_acc, conmx, val_loss = comp_test('Validation', clf, testloader, criterion, disp)
        records['valid'].append([cur_acc, conmx, val_loss])
        
        if(disp):
            print('validation loss: ', val_loss)

        scheduler.step(val_loss)
        
        if max_test_perf - val_loss > min_delta:
            if(disp):
                print('refresh patience')
            max_test_perf = val_loss
            counter = 0
            # save model
            cur_high = [cur_acc, conmx]
            torch.save(clf.state_dict(), save_path)
    #                 print('after  val_loss', val_loss, 'best_loss', best_loss)
        elif max_test_perf - val_loss < min_delta:
#             if (episode > 50):
            if(disp):
                print('patience counter +1')
            counter += 1
            if counter >= patience:
                break

    # print('\t'.join([str(it) for it in [cur_high[3], cur_high[0], cur_high[1], cur_high[2]]]))


    if(disp):
        print('Finished Training')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)

    return records

def run_finetune(model_name, train_csv, val_csv, root_folder, model_path, save_path, disp, nb_cls=6, batch_size=10, lr=0.0001, patience=5, min_delta=0, max_episodes=1000):
    start_time = time.time()

    test_csv = val_csv

    train_dataset = tiny_Dataset(csv_file=train_csv,
                                 root_dir=root_folder,
                                 transform=transforms.Compose([
                                     Rescale((224,224)),
                                     ToTensor(),
                                     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                 ]))
    test_dataset = tiny_Dataset(csv_file=test_csv,
                                root_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    clf = get_pretrain_model(model_name, nb_cls)
    clf.to(device)
    clf.load_state_dict(torch.load(model_path))

    # pre-defined loss weights based on preliminary experiments
    # 1/class_precision
    cls_weights = torch.tensor([1.46993504, 1.83937636, 1.63301425, 1.10534349, 1., 1.]).to(device)

    criterion = nn.BCEWithLogitsLoss(weight=cls_weights)
    optimizer_clf = optim.AdamW(clf.parameters(), lr=lr)

    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_clf, mode='min', factor=0.5, patience=3, verbose=disp, min_lr=5e-7
    )

    max_test_perf = 10000
    min_delta = min_delta
    patience = patience
    counter = 0

    MAX_EPISODES = max_episodes
    PRINT_EVERY = 1

    records = {'train': [],'valid': []}
    for episode in range(1, MAX_EPISODES+1):  # loop over the dataset multiple times
        if(disp):
            print('episode:', episode)
        critic_loss = train(clf, optimizer_clf, trainloader, criterion, disp)
        if(disp):
            print('Train')
        tr_cur_acc, tr_conmx, tr_loss = comp_test('Train', clf, trainloader, criterion, disp)
        records['train'].append([tr_cur_acc, tr_conmx, tr_loss])
        if(disp):
            print('train loss: ', critic_loss)
            print('train loss: ', tr_loss)
            print('Validation')
        cur_acc, conmx, val_loss = comp_test('Validation', clf, testloader, criterion, disp)
        records['valid'].append([cur_acc, conmx, val_loss])
        
        if(disp):
            print('validation loss: ', val_loss)

        scheduler.step(val_loss) # step the scheduler based on validation loss

        if max_test_perf - val_loss > min_delta:
            if(disp):
                print('refresh patience')
            max_test_perf = val_loss
            counter = 0
            # save model
            cur_high = [cur_acc, conmx]
            torch.save(clf.state_dict(), save_path)
    #                 print('after  val_loss', val_loss, 'best_loss', best_loss)
        elif max_test_perf - val_loss < min_delta:
#             if (episode > 50):
            if(disp):
                print('patience counter +1')
            counter += 1
            if counter >= patience:
                break

def run_test(model_name, test_csv, root_folder, model_path, disp, nb_cls=6, batch_size=10):
    # run test
    start_time = time.time()

    pth = model_path
    
    test_dataset = tiny_Dataset(csv_file=test_csv,
                                root_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor(),
                                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    clf = get_pretrain_model(model_name, nb_cls)
    clf.load_state_dict(torch.load(pth))
    clf.to(device)

    # pre-defined loss weights based on preliminary experiments
    # 1/class_precision
    cls_weights = torch.tensor([1.46993504, 1.83937636, 1.63301425, 1.10534349, 1., 1.]).to(device)

    criterion = nn.BCEWithLogitsLoss(weight=cls_weights)
    cur_acc, conmx, val_loss = comp_test('Test', clf, testloader, criterion, disp)

    if(disp):
        print('Finished Testing')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)
    
    return cur_acc, conmx, val_loss

def prototypical_loss(query_embeddings, query_labels, prototypes):
    # prototypes: [n_classes, feat_dim]
    # query_embeddings: [batch_size, feat_dim]
    # query_labels: [batch_size]
    dists = torch.cdist(query_embeddings, prototypes)  # [B, N]
    log_p_y = F.log_softmax(-dists, dim=1)
    loss = F.nll_loss(log_p_y, query_labels.long())
    preds = log_p_y.argmax(dim=1)
    return loss, preds

def run_train_protonet(model_name, train_csv, anchor_csv, val_csv, root_folder, save_path, disp, nb_cls=6, batch_size=10, lr=0.0001, patience=5, min_delta=0, max_episodes=1000):
    start_time = time.time()

    test_csv = val_csv

    train_dataset = ProtoDataset(csv_file=train_csv, root_dir=root_folder,
                                 anchor_csv_file=anchor_csv, anchor_dir=root_folder,
                                 transform=transforms.Compose([
                                     Rescale((224,224)),
                                     ToTensor()
                                 ]),
                                 nb_cls=nb_cls)
    
    test_dataset = ProtoDataset(csv_file=test_csv, root_dir=root_folder,
                                anchor_csv_file=anchor_csv, anchor_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor()
                                ]),
                                nb_cls=nb_cls)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)

    clf = get_pretrain_model(model_name, nb_cls) # protonet_convnext_base
    clf.to(device)

    optimizer_clf = optim.AdamW(clf.parameters(), lr=lr)

    max_test_perf = 10000
    min_delta = min_delta
    patience = patience
    counter = 0

    MAX_EPISODES = max_episodes
    PRINT_EVERY = 1

    records = {'train': [],'valid': []}
    for episode in range(1, MAX_EPISODES+1):  # loop over the dataset multiple times
        if(disp):
            print('Episode:', episode)
        clf.train()

        # allow gradients for prototypes
        prototypes = []
        for cls in range(nb_cls):
            # print(train_dataset.support[cls].shape)
            support_images = train_dataset.support[cls].unsqueeze(0).to(device)
            proto = clf(support_images)
            prototypes.append(proto.squeeze(0))
        prototypes = torch.stack(prototypes).to(device)

        # alternatively, do not allow gradients for prototypes
        # with torch.no_grad():
        #     clf.eval()
        #     prototypes = []
        #     for cls in range(nb_cls):
        #         # print(train_dataset.support[cls].shape)
        #         support_images = train_dataset.support[cls].unsqueeze(0).to(device)
        #         proto = clf(support_images)
        #         prototypes.append(proto.squeeze(0))
        #     prototypes = torch.stack(prototypes).to(device)

        train_loss, train_conmx, train_acc = train_protonet(
            trainloader=trainloader,
            clf=clf,
            optimizer_clf=optimizer_clf,
            prototypes=prototypes,
            disp=disp
        )

        if(disp):
            print('Episode Train Done')
            print('train loss: ', train_loss)
            print('Train accuracy: ', train_acc)
        records['train'].append([train_acc, train_conmx, train_loss])

        val_acc, val_conmx, validation_loss = evaluate_model(
            clf=clf,
            testloader=testloader,
            prototypes=prototypes,
            device=device,
            disp=disp
        )

        records['valid'].append([val_acc, val_conmx, validation_loss])
        if(disp):
            print('Episode alidation Done')
            print('validation loss: ', validation_loss)
            print('Validation accuracy: ', val_acc)

        if max_test_perf - validation_loss > min_delta:
            if(disp):
                print('refresh patience')
            max_test_perf = validation_loss
            counter = 0
            # save model
            cur_high = [val_acc, val_conmx]
            torch.save(clf.state_dict(), save_path)

        elif max_test_perf - validation_loss < min_delta:
            if(disp):
                print('patience counter +1')
            counter += 1
            if counter >= patience:
                break

    if(disp):
        print('Finished Training')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print('Total training time: ', time_elapsed)

    return records

def run_test_protonet(model_name, test_csv, anchor_csv, root_folder, model_path, disp, nb_cls=6, batch_size=10):
    # run test
    start_time = time.time()

    pth = model_path
    
    test_dataset = ProtoDataset(csv_file=test_csv, root_dir=root_folder,
                                anchor_csv_file=anchor_csv, anchor_dir=root_folder,
                                transform=transforms.Compose([
                                    Rescale((224,224)),
                                    ToTensor()
                                ]),
                                nb_cls=nb_cls)
    
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    clf = get_pretrain_model(model_name, nb_cls)  # protonet_convnext_base
    clf.load_state_dict(torch.load(pth))
    clf.to(device)

    with torch.no_grad():
        clf.eval()
        prototypes = []
        for cls in range(nb_cls):
            support_images = test_dataset.support[cls].unsqueeze(0).to(device)
            proto = clf(support_images)
            prototypes.append(proto.squeeze(0))
        prototypes = torch.stack(prototypes).to(device)

    val_acc, conmx, validation_loss = evaluate_model(
        clf=clf,
        testloader=testloader,
        prototypes=prototypes,
        device=device,
        disp=disp
    )

    if(disp):
        print('Finished Testing')
        print('Test accuracy: ', val_acc)
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print('Total testing time: ', time_elapsed)
    
    return val_acc, conmx, validation_loss

def train_protonet(trainloader, clf, optimizer_clf, prototypes, disp=False):
    clf.train()
    correct = 0
    total = 0
    count = 0
    value_losses = []
    preds = np.empty(0)
    lbs = np.empty(0)
    if(disp):
        print(device)
    for i, data in enumerate(trainloader, 0):
        count += 1
        # print(data[0].shape, data[1].shape)
        if device is None:
            inputs = data[0].type(torch.FloatTensor)
            labels = data[1].type(torch.FloatTensor).squeeze()
        else:
            inputs = data[0].type(torch.FloatTensor).to(device)
            labels = data[1].type(torch.FloatTensor).squeeze().to(device)

        feats = clf(inputs)
            
        value_loss, value_pred = prototypical_loss(feats, labels, prototypes)

    #         if(disp):
    #             print(value_loss)
            
        optimizer_clf.zero_grad()
        value_loss.backward()
        optimizer_clf.step()
            
        value_losses.append(float(value_loss.item()))

        value_pred = value_pred.detach().cpu().numpy()
        total += labels.size(0)
        lb_npy = labels.detach().cpu().numpy()
        correct += (value_pred == lb_npy).sum().item()
        preds = np.hstack((preds, value_pred.squeeze()))
        lbs = np.hstack((lbs, lb_npy.squeeze()))

    conmx = confusion_matrix(lbs, preds)
    acc = correct / total

    train_loss = sum(value_losses)/len(value_losses)
    return train_loss, conmx, acc

def evaluate_model(testloader, clf, prototypes, device, disp=False):
    """
    Evaluate the model on the test dataset.
    
    Args:
        clf: The model to evaluate.
        testloader: DataLoader for the test dataset.
        prototypes: Precomputed prototypes for prototypical networks.
        device: Device to run the evaluation on (e.g., 'cuda' or 'cpu').
        disp: Whether to display additional information.

    Returns:
        acc: Accuracy of the model on the test dataset.
        conmx: Confusion matrix.
        validation_loss: Average validation loss.
    """
    with torch.no_grad():
        clf.eval()
        correct = 0
        total = 0
        preds = np.empty(0)
        lbs = np.empty(0)
        loss = []

        if disp:
            print(device)

        for data in testloader:
            inputs = data[0].to(device) if device else data[0]
            labels = data[1].to(device).squeeze() if device else data[1].squeeze()

            val_feats = clf(inputs)
            val_loss, predicted = prototypical_loss(val_feats, labels, prototypes)
            loss.append(val_loss.item())

            predicted = predicted.detach().cpu().numpy()
            total += labels.size(0)
            lb_npy = labels.detach().cpu().numpy()
            correct += (predicted == lb_npy).sum().item()
            preds = np.hstack((preds, predicted.squeeze()))
            lbs = np.hstack((lbs, lb_npy.squeeze()))

        conmx = confusion_matrix(lbs, preds)
        acc = correct / total

        validation_loss = sum(loss) / len(loss)

    return acc, conmx, validation_loss

def run(var_save_name, model_name, model_save_path,
        train_csv, valid_csv, test_csv, base_path,
        run_count = 1, disp = False, 
        nb_cls=6, batch_size=10, lr=0.0001, patience=5, min_delta=0, max_episodes=1000):
    
    exps_rslts = []
    for iter_count in range(run_count):

        train_records = run_train(model_name,
                                  train_csv,
                                  valid_csv,
                                  base_path,
                                  model_save_path,
                                  disp,
                                  nb_cls=nb_cls,
                                  batch_size=batch_size,
                                  lr=lr,
                                  patience=patience,
                                  min_delta=min_delta,
                                  max_episodes=max_episodes
                                  )
        print(var_save_name, 'train')
        cur_acc, conmx, val_loss = run_test(model_name,
                                            test_csv,
                                            base_path,
                                            model_save_path,
                                            disp
                                            )
        print(var_save_name, 'test')
        exps_rslts.append([cur_acc, conmx, val_loss, train_records])
        print(cur_acc)
        print(conmx)

        with open('result_data_'+var_save_name+'_'+str(iter_count)+'.pkl', 'wb') as fp:
            pickle.dump(exps_rslts, fp)
            print('exps rslts saved successfully to file: ', iter_count)
        
#         print(var_save_name, iter_count)
#         print(exps_rslts)

def run_hybrid(var_save_name, model_name, pretrain_model_weight_path, model_save_path,
               train_csv, valid_csv, test_csv, base_path,
               run_count = 1, disp = False, 
               nb_cls=6, batch_size=10, lr=0.0001, patience=5, min_delta=0, max_episodes=1000):

    exps_rslts = []
    for iter_count in range(run_count):

        train_records = run_finetune(model_name,
                                     train_csv,
                                     valid_csv,
                                     base_path,
                                     pretrain_model_weight_path,
                                     model_save_path,
                                     disp, 
                                     nb_cls=nb_cls, 
                                     batch_size=batch_size, 
                                     lr=lr, 
                                     patience=patience, 
                                     min_delta=min_delta, 
                                     max_episodes=max_episodes
                                    )
        print(var_save_name, 'train')
        cur_acc, conmx, val_loss = run_test(model_name,
                                            test_csv,
                                            base_path,
                                            model_save_path,
                                            disp
                                            )
        print(var_save_name, 'test')
        exps_rslts.append([cur_acc, conmx, val_loss, train_records])
        print(cur_acc)
        print(conmx)

        with open('result_data_'+var_save_name+'_'+str(iter_count)+'.pkl', 'wb') as fp:
            pickle.dump(exps_rslts, fp)
            print('exps rslts saved successfully to file: ', iter_count)


def run_protonet(var_save_name, model_name, model_save_path,
                 train_csv, anchor_csv, valid_csv, test_csv, base_path,
                 run_count = 1, disp = False, 
                 nb_cls=6, batch_size=10, lr=0.0001, patience=5, min_delta=0, max_episodes=1000):
    
    exps_rslts = []
    for iter_count in range(run_count):

        train_records = run_train_protonet(model_name,
                                           train_csv,
                                           anchor_csv,
                                           valid_csv,
                                           base_path,
                                           model_save_path,
                                           disp,
                                           nb_cls=nb_cls,
                                           batch_size=batch_size,
                                           lr=lr,
                                           patience=patience,
                                           min_delta=min_delta,
                                           max_episodes=max_episodes
                                          )
        
        cur_acc, conmx, val_loss = run_test_protonet(model_name,
                                                     test_csv,
                                                     anchor_csv,
                                                     base_path,
                                                     model_save_path,
                                                     disp,
                                                     nb_cls=nb_cls,
                                                     batch_size=batch_size
                                                    )
        
        exps_rslts.append([cur_acc, conmx, val_loss, train_records])

        with open('result_data_'+var_save_name+'_'+str(iter_count)+'.pkl', 'wb') as fp:
            pickle.dump(exps_rslts, fp)
            print('exps rslts saved successfully to file: ', iter_count)
