import os
import shutil
import argparse
import yaml
from easydict import EasyDict
from tqdm.auto import tqdm
from glob import glob
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import DataLoader
from torch_geometric.data import Batch
from models.epsnet import get_model
from utils.datasets import ConformationDataset
from utils.transforms import *
from utils.misc import *
from utils.common import get_optimizer, get_scheduler, get_inner_optimizer
from copy import deepcopy
import higher

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=str, default=None) #here I changed type to str to allow for the named checkpoints
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--maml', type=int, default=1)
    parser.add_argument('--val_freq', type=float, default=1) 
    parser.add_argument('--val_size', type=float, default=1) #do on a subset of validation set
    parser.add_argument('--max_iter', type=int, default=10000000000)
    parser.add_argument('--inner_loop_steps', type=int, default=1)
    parser.add_argument('--outer_batch_size', type=int, default=32)
    parser.add_argument('--curriculum', type=int, default=1)
    parser.add_argument('--learn_inner_lr', type=int, default=1)
    parser.add_argument('--scale_ilr', type=int, default=1)
    parser.add_argument('--scale_olr', type=int, default=1)
    args = parser.parse_args()
    print(args.config)
    resume = os.path.isdir(args.config)
    print(resume)
    if resume:
        config_path = glob(os.path.join(args.config, '*.yml'))[0]
        resume_from = args.config
    else:
        config_path = args.config

    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)

    
    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume', args=args)
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, args=args)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    print(log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    if args.curriculum == True:
        #reorder dataset from low to high number of atoms
        num_atom_array = []
        for i in range(len(train_set)):
            num_atom_array.append(train_set[i].num_nodes)
        train_set.data = [y for _,y in sorted(zip(num_atom_array, train_set.data), key=lambda y: y[0])]
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    num_examples_per_task = 5
    train_iterator = inf_iterator(DataLoader(train_set, num_examples_per_task, shuffle=False)) 
    val_loader = DataLoader(val_set, num_examples_per_task, shuffle=False)

    # Model
    logger.info('Building model...')
    model = get_model(config.model, config.train.optimizer.lr/args.scale_ilr).to(args.device)
    # Optimizer
    separate_opt = False
    if separate_opt == True:
        optimizer_global = get_optimizer(config.train.optimizer, model.model_global)
        optimizer_local = get_optimizer(config.train.optimizer, model.model_local)
        scheduler_global = get_scheduler(config.train.scheduler, optimizer_global)
        scheduler_local = get_scheduler(config.train.scheduler, optimizer_local)
    else:
        config.train.optimizer.lr = config.train.optimizer.lr/args.scale_olr
        if args.learn_inner_lr == True:
            optimizer = get_optimizer(config.train.optimizer, list(model.model_global.parameters()) + list(model.model_local.parameters()) + list(model.inner_lrs.values()))
        else:
            optimizer = get_optimizer(config.train.optimizer, list(model.model_global.parameters()) + list(model.model_local.parameters()))
        config.train.optimizer.lr = config.train.optimizer.lr*args.scale_olr
        inner_opt = get_inner_optimizer(config.train.optimizer, model)
        #scheduler = get_scheduler(config.train.scheduler, optimizer)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    start_iter = 1

    # Resume from checkpoint
    if resume:
        print(resume_from)
        ckpt_path, start_iter = get_checkpoint_path(os.path.join(resume_from, 'checkpoints'), it=args.resume_iter)
        logger.info('Resuming from: %s' % ckpt_path)
        if type(start_iter) == str:
            logger.info('Iteration: %s' % start_iter)
        elif type(start_iter) == int:
            logger.info('Iteration: %d' % start_iter)
        start_iter = 1 #now start fine-tuning, just a label from now on
        if args.device == "cpu":
            ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
        else:
            ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        if separate_opt == True:
            optimizer_global.load_state_dict(ckpt['optimizer_global'])
            optimizer_local.load_state_dict(ckpt['optimizer_local'])
            scheduler_global.load_state_dict(ckpt['scheduler_global'])
            scheduler_local.load_state_dict(ckpt['scheduler_local'])
        else:
            #MY CONTRIBUTION
            config.train.optimizer.lr = config.train.optimizer.lr/args.scale_olr
            if args.learn_inner_lr == True:
                optimizer = get_optimizer(config.train.optimizer, list(model.model_global.parameters()) + list(model.model_local.parameters()) + list(model.inner_lrs.values()))
            else:
                optimizer = get_optimizer(config.train.optimizer, list(model.model_global.parameters()) + list(model.model_local.parameters()))
            config.train.optimizer.lr = config.train.optimizer.lr*args.scale_olr
            inner_opt = get_inner_optimizer(config.train.optimizer, model)
            #scheduler = get_scheduler(config.train.scheduler, optimizer)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=5)
    #import pdb; pdb.set_trace()
    def inner_loop(model, inner_opt, task, num_inner_loop_steps):
        #MY CONTRIBUTION
        """
        Idea: make a copy of the model, update parameters of the copied model for num_inner_loop_steps

        Input: model (instance of the NN), inner_opt (optimizer for inner loop), task (Batch object with support and query set)
        Returns: loss on query after adaptation (it's a 2D tensor, output of GeoDiff)
        """
        data_list = task.to_data_list()

        support = Batch.from_data_list(data_list[:(num_examples_per_task-1)])  #needs to be a batch object with only support set
        query = Batch.from_data_list(data_list[(num_examples_per_task-1):])  #also a batch object with only query set
        inner_opt.zero_grad()

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            if args.learn_inner_lr == True:
                for name, g in zip(fnet.param_names, diffopt.param_groups): #have to make sure both param_names and param_groups etc in same order
                    g['lr'] = fnet.inner_lrs[name]

            for _ in range(num_inner_loop_steps):
                support_loss, loss_global, loss_local = fnet.get_loss(
                    atom_type=support.atom_type,
                    pos=support.pos,
                    bond_index=support.edge_index,
                    bond_type=support.edge_type,
                    batch=support.batch,
                    num_nodes_per_graph=support.num_nodes_per_graph,
                    num_graphs=support.num_graphs,
                    anneal_power=config.train.anneal_power,
                    return_unreduced_loss=True
                )
                support_loss = support_loss.mean()
                
                diffopt.step(support_loss)
            query_loss, loss_global, loss_local = fnet.get_loss(
                atom_type=query.atom_type,
                pos=query.pos,
                bond_index=query.edge_index,
                bond_type=query.edge_type,
                batch=query.batch,
                num_nodes_per_graph=query.num_nodes_per_graph,
                num_graphs=query.num_graphs,
                anneal_power=config.train.anneal_power,
                return_unreduced_loss=True
            )
            #print("query_loss mean:", query_loss)
        return query_loss

    def train(it):
        model.train()
        # optimizer_global.zero_grad()
        # optimizer_local.zero_grad()
        optimizer.zero_grad()
        MAML = args.maml
        if MAML == True:
            #MY CONTRIBUTION
            num_inner_loop_steps = args.inner_loop_steps #experiment with this
            batch_size_outer_loop = args.outer_batch_size #experiment with this
            #compute inner loop each time, and then compute outer loop
            #in inner loop adapt parameters manually, then do loss.backward for outer loop
            task_batch = []
            for _ in range(batch_size_outer_loop):
                task_batch.append(next(train_iterator).to(args.device))
            outer_loss_batch = []
            for task in task_batch:
                query_loss = inner_loop(model, inner_opt, task, num_inner_loop_steps)
                query_loss = query_loss.mean()
                query_loss.backward()
                outer_loss_batch.append(query_loss.detach())
            loss = torch.mean(torch.stack(outer_loss_batch))
            optimizer.step()
            #print(model.inner_lrs)
            
        else:
        #previous approach
            batch = next(train_iterator).to(args.device)
            loss, loss_global, loss_local = model.get_loss(
                atom_type=batch.atom_type,
                pos=batch.pos,
                bond_index=batch.edge_index,
                bond_type=batch.edge_type,
                batch=batch.batch,
                num_nodes_per_graph=batch.num_nodes_per_graph,
                num_graphs=batch.num_graphs,
                anneal_power=config.train.anneal_power,
                return_unreduced_loss=True
            )
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            # optimizer_global.step()
            # optimizer_local.step()
        
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        
        logger.info('[Train] Iter %05d | Loss %.2f | Grad %.2f | LR %.6f' % (
            it, loss.item(), orig_grad_norm, optimizer.param_groups[0]['lr'],
        ))
        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad_norm', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = 0, 0
        sum_loss_global, sum_n_global = 0, 0
        sum_loss_local, sum_n_local = 0, 0 
        model.train()
        MAML = args.maml
        if MAML == True:
            #MY CONTRIBUTION
            num_inner_loop_steps = args.inner_loop_steps #experiment with this
            val_size = int(args.val_size*len(val_set))
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                if i >= val_size:
                    break
                batch = batch.to(args.device)
                loss_batch = []
                query_loss = inner_loop(model, inner_opt, batch, num_inner_loop_steps) #batch here is just one task
                loss_batch.append(query_loss.detach())
            avg_loss = torch.mean(torch.stack(loss_batch)) 
        else:
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch = batch.to(args.device)
                #only do validation on query
                batch = Batch.from_data_list(batch.to_data_list()[(num_examples_per_task-1):])  #also a batch object with only query set
                loss, loss_global, loss_local = model.get_loss(
                    atom_type=batch.atom_type,
                    pos=batch.pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_nodes_per_graph=batch.num_nodes_per_graph,
                    num_graphs=batch.num_graphs,
                    anneal_power=config.train.anneal_power,
                    return_unreduced_loss=True
                )
                sum_loss += loss.sum().item()
                sum_n += loss.size(0)
                # sum_loss_global += loss_global.sum().item()
                # sum_n_global += loss_global.size(0)
                # sum_loss_local += loss_local.sum().item()
                # sum_n_local += loss_local.size(0)
            avg_loss = sum_loss / sum_n
            # avg_loss_global = sum_loss_global / sum_n_global
            # avg_loss_local = sum_loss_local / sum_n_local
        
        if config.train.scheduler.type == 'plateau':
            # scheduler_global.step(avg_loss_global)
            # scheduler_local.step(avg_loss_local)
            scheduler.step(avg_loss)
        else:
            # scheduler_global.step()
            # scheduler_local.step()
            scheduler.step()

        # logger.info('[Validate] Iter %05d | Loss %.6f | Loss(Global) %.6f | Loss(Local) %.6f' % (
        #     it, avg_loss, avg_loss_global, avg_loss_local,
        # ))
        logger.info('[Validate] Iter %05d | Loss %.6f' % (
            it, avg_loss,
        ))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss

    try:
        max_iter = args.max_iter
        val_freq = int(len(train_set)*args.val_freq/(args.outer_batch_size*num_examples_per_task))
        for it in range(start_iter, max_iter + 1):
            train(it)
            if it % val_freq == 0 or it == max_iter:
                avg_val_loss = validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                    'avg_val_loss': avg_val_loss,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')

