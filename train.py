import argparse 
import os 
import numpy as np 
import logging 
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable 
import torch.optim as optim 
import json 
import models.SVHNet as svhnet
import utils
from tqdm import tqdm
from eval import evaluate
from dataloader_utils import load_dataset

"""
A script to train models using checkpointing and the evaluation at fixed steps on val set. 
The machinery has been borrowed from cs230 @ Stanford University. 
"""
parser = argparse.ArgumentParser()
parser.add_argument('--param_path', default=None, help="Path to the folder having params.json")
parser.add_argument('--resume_path', default=None, help='Path to any previous saved checkpoint')
parser.add_argument('--teacher_checkpoint', default=None, help='Full Path to a trained teacher model')


def train(net, dataloader, loss_fn, params, metrics, optimizer):
    """
    Train the net for one epoch i.e 1..len(dataloader)
    net: The model to test
    params: The hyperparams 
    loss_fn: The loss function
    metrics: The metrics dictionary containing evaluation metrics. 
    """
    net.train()

    summaries = [] 
    loss_avg = utils.AverageMeter()
    with tqdm(total=len(dataloader)) as t:
        for i, (data_batch, label_batch) in enumerate(dataloader):
            if params.cuda:
                data_batch, label_batch = data_batch.cuda(), label_batch.cuda()
            
            data_batch, label_batch = Variable(data_batch), Variable(label_batch)

            # print(data_batch.size())
            # print(label_batch.size())

            output_batch = net(data_batch)
            print(output_batch.size())
            loss = loss_fn(output_batch, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % params.save_summary_steps == 0 :
                out_np = output_batch.data.cpu().numpy()
                label_np = label_batch.data.cpu().numpy()
                batch_summary = {metric: metrics[metric](out_np, label_np) for metric in metrics}
                batch_summary['loss'] = loss.data[0].cpu().item()
                summaries.append(batch_summary)
            
            loss_avg.update(loss.data[0].cpu().item())
            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
    
    # compute mean of all the metrics

    mean_metrics = {metric:np.mean([m[metric] for m in summaries]) for metric in summaries[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in mean_metrics.items())
    logging.info("Train Metrics: "+ metrics_string)


def train_and_eval(net, train_loader, val_loader, optimizer, loss_fn, metrics, params, model_dir, restore=None):
    """
    Train and evaluate every epoch of a model.
    net: The model. 
    train/val loader: The data loaders
    params: The parameters parsed from JSON file 
    restore: if there is a checkpoint restore from that point. 
    """
    best_val_acc = 0.0 
    if restore is not None:
        restore_file = os.path.join(args.param_path, args.resume_path + '_pth.tar')
        logging.info("Loaded checkpoints from:{}".format(restore_file))
        utils.load_checkpoint(restore_file, net, optimizer)

    for ep in range(params.num_epochs):
        logging.info("Running epoch: {}/{}".format(ep+1, params.num_epochs))

        # train one epoch 
        train(net, train_loader, loss_fn, params, metrics, optimizer)

        val_metrics = evaluate(net, val_loader, loss_fn, params, metrics)

        val_acc = val_metrics['accuracy']
        isbest = val_acc >= best_val_acc 

        utils.save_checkpoint({"epoch":ep, "state_dict":net.state_dict(), "optimizer":optimizer.state_dict()}, 
        isBest=isbest, ckpt_dir=model_dir)
    
        if isbest:
            # if the accuracy is great  save it to best.json 
            logging.info("New best accuracy found!")
            best_val_acc = val_acc 
            best_json_path = os.path.join(model_dir, "best_model_params.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
        
        last_acc_path = os.path.join(model_dir, 'last_acc_metrics.json')
        utils.save_dict_to_json(val_metrics, last_acc_path)








if __name__ == '__main__':
    args = parser.parse_args()
    params = utils.ParamParser(os.path.join(args.param_path, 'params.json'))

    params.cuda = torch.cuda.is_available()

    utils.setLogger(os.path.join(args.param_path, "train.log"))

    logging.info("Loading the datasets")


    # TODO: Load the datasets 
    train_loader = load_dataset('train', 'cifar',params)
    val_loader = load_dataset('val', 'cifar', params)

    logging.info("finished loading the datasets") 

    if params.model_name == "base":
        model = svhnet.BaseSVHNet(params.initial_channel, params.kernel_size, use_dropout=True).cuda() if params.cuda else svhnet.BaseSVHNet(params.initial_channel, params.kernel_size, use_dropout=True)
    elif params.model_name == "stn":
        spatial_dim = (params.height, params.width)
        model = svhnet.STNSVHNet(spatial_dim, params.initial_channel, params.stn_kernel_size, params.kernel_size, use_dropout=True).cuda() if params.cuda else svhnet.STNSVHNet(spatial_dim, params.initial_channel, params.stn_kernel_size, params.kernel_size, use_dropout=True)

    loss_fn = svhnet.loss_fn
    metrics = svhnet.metrics
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    logging.info("Started training for {} epochs".format(params.num_epochs))
    train_and_eval(model, train_loader, val_loader, optimizer,loss_fn, metrics, params, args.param_path)
        

    # loss_fn = net.loss_function 
    # metrics = net.metrics 

    # if params.model_name == "resnet18":
    #     model = resnet.ResNet18().cuda() if params.cuda else resnet.ResNet18(params)
    # elif params.model_name == "cnn":
    #     model = net.CIFARNet(params).cuda() if params.cuda else net.CIFARNet(params) 

    
    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    
    # train_and_eval(model, train_loader, val_loader, optimizer,loss_fn, metrics, params, args.param_path)
     






        


        
            





