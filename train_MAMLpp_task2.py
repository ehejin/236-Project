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
import spacial_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume_iter', type=str, default=None) #here I changed type to str to allow for the named checkpoints
    parser.add_argument('--train_set',type=str, default="drugs")
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--maml', type=int, default=1)
    parser.add_argument('--val_freq', type=float, default=1) 
    parser.add_argument('--val_size', type=float, default=1) #do on a subset of validation set
    parser.add_argument('--max_iter', type=int, default=10050)
    parser.add_argument('--inner_loop_steps', type=int, default=1)
    parser.add_argument('--outer_batch_size', type=int, default=4)
    parser.add_argument('--curriculum', type=int, default=1)
    parser.add_argument('--learn_inner_lr', type=int, default=1)
    parser.add_argument('--scale_ilr', type=int, default=1)
    parser.add_argument('--scale_olr', type=int, default=1)
    parser.add_argument('--sup_size', type=int, default=4)
    parser.add_argument('--decrease_sup_size', type=int, default=0)
    parser.add_argument('--msl', type=int, default=1)
    parser.add_argument('--curriculum_type', type=int, default=0)
    parser.add_argument('--size_tr_MAML', type=int, default=1)
    args = parser.parse_args()
    MAML = args.maml
    print(args.config)
    resume = os.path.isdir(args.config)
    if resume:
        config_path = args.config + args.resume_iter + ".yml"
        resume_from = args.config
        config_path_train = args.config + args.train_set+"_default.yml"
        with open(config_path, 'r') as f:
            config_pretrained = EasyDict(yaml.safe_load(f))
        with open(config_path_train, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
    else:
        config_path = args.config
        with open(config_path, 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        config_pretrained = config

    
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    seed_all(config.train.seed)
    
    # Logging
    if resume:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, tag='resume', args=args)
        os.symlink(os.path.realpath(resume_from), os.path.join(log_dir, os.path.basename(resume_from.rstrip("/"))))
    else:
        log_dir = get_new_log_dir(args.logdir, prefix=config_name, args=args)
        shutil.copytree('./models', os.path.join(log_dir, 'models'))
    os.rename(log_dir, log_dir+'_tt_2')
    log_dir = log_dir + '_tt_2'
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(config_path, os.path.join(log_dir, os.path.basename(config_path)))

    # Datasets and loaders
    
    logger.info('Loading datasets...')
    transforms = CountNodesPerGraph()
    train_set = ConformationDataset(config.dataset.train, transform=transforms)
    num_examples_per_task = 5

    if args.curriculum == False:
        train_iterator_geodiff = inf_iterator(DataLoader(train_set, num_examples_per_task, shuffle=True)) #bc now no task, so just random

    #curriculum_type=0: successive relearning
    #curriculum_type=1: split 1 epoch into say, 10 steps. Then multiply each split by number of 
    #curriculum_type=2: split each section in say, 10 steps. Then for each step, sample batches from a certain percentage of the data

    num_tasks = 10 #split dataset into 10 tasks based on size
    epoch_size = int(len(train_set)/(num_examples_per_task*args.outer_batch_size))
    size_train_MAML = args.size_tr_MAML #in epochs
    
    size_train_geodiff = int(args.max_iter/epoch_size)-size_train_MAML #in epochs
    train_loader = []
    num_atom_array = []
    for i in range(len(train_set)):
        num_atom_array.append(train_set[i].num_nodes)
    train_set.data = [y for _,y in sorted(zip(num_atom_array, train_set.data), key=lambda y: y[0])] #by inspection it preserves groupings
    idx_lists = []
    for i in range(num_tasks):
        idx_lists.append(np.arange(int(len(train_set)*(i)/num_tasks),int(len(train_set)*(i+1)/num_tasks)))
    if args.curriculum == True:
        #now here have to use other complexity metric to sort how to sample from tasks
        #sort each individual task by molecular complexity
        mol_complexity_array = []
        for data in tqdm(train_set.data):
            mol_complexity_array.append(spacial_score.calculate_score_from_smiles(data.smiles))
        
        tr_data_copy = np.empty(len(train_set), dtype=object)
        for i in range(len(train_set)):
            tr_data_copy[i] = train_set.data[i]
        l = len(tr_data_copy)
        tr_data_copy = np.reshape(tr_data_copy, (num_tasks, -1))
        
        for i in range(len(tr_data_copy)):
            tr_data_copy[i] = [y for _,y in sorted(zip(mol_complexity_array[int((i/num_tasks)*l):int(((i+1)/num_tasks)*l)], tr_data_copy[i]), key=lambda y: y[0])]
        
        train_set.data = list(tr_data_copy.flatten())
        if args.curriculum_type == 0:
            raise NotImplementedError()
            #train_iterator = inf_iterator(DataLoader(train_set, num_examples_per_task, shuffle=False)) 

        elif args.curriculum_type == 1:
            raise NotImplementedError()
            # train_set.data = list(np.tile(np.reshape(tr_data_copy, (epoch_split, -1)), size_train_MAML).flatten())
            # train_iterator = inf_iterator(DataLoader(train_set, num_examples_per_task, shuffle=False))
        elif args.curriculum_type == 2:
            #split task, and then at each iteration sample from that 
            task_split = 10
            starting_pct = 0.1 #start with 10% of task
            idx_bound_tasks = []
            step_length = epoch_size/task_split
            inc = round((1/starting_pct)**(1/(task_split-1)),3)+0.001

            for i in range(task_split):
                idx_bound_tasks.append(int((l/num_tasks)*min(starting_pct*inc**(i),1)))
    val_set = ConformationDataset(config.dataset.val, transform=transforms)
    val_data_copy = np.empty(len(val_set), dtype=object)
    for i in range(len(val_set)):
        val_data_copy[i] = val_set.data[i]
    val_data_copy = np.reshape(val_data_copy, (num_tasks, -1))
    for i in range(len(val_data_copy)):
        random.shuffle(val_data_copy[i])
    val_set.data = list(val_data_copy.flatten())
    val_loader = DataLoader(val_set, num_examples_per_task, shuffle=False)

    # Model
    logger.info('Building model...')
    #load pretrained model, but train on different
    model = get_model(config_pretrained.model, config_pretrained.train.optimizer.lr/args.scale_ilr).to(args.device)
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
    it_geodiff = 1
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
    def get_per_step_loss_importance_vector(current_step):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(args.inner_loop_steps)) * (
                1.0 / args.inner_loop_steps)
        reference_step_number = int(len(train_set)/(args.outer_batch_size*num_examples_per_task))
        decay_rate = 1.0 / args.inner_loop_steps / reference_step_number
        min_value_for_non_final_losses = 0.03 / args.inner_loop_steps
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (current_step * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (current_step * (args.inner_loop_steps - 1) * decay_rate),
            1.0 - ((args.inner_loop_steps - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=args.device)
        return loss_weights
    
    def inner_loop(model, inner_opt, task, num_inner_loop_steps, it):
        #MY CONTRIBUTION
        """
        Idea: make a copy of the model, update parameters of the copied model for num_inner_loop_steps

        Input: model (instance of the NN), inner_opt (optimizer for inner loop), task (Batch object with support and query set)
        Returns: loss on query after adaptation (it's a 2D tensor, output of GeoDiff)
        """
        data_list = task.to_data_list()

        support = Batch.from_data_list(data_list[:args.sup_size])  #needs to be a batch object with only support set
        query = Batch.from_data_list(data_list[args.sup_size:])  #also a batch object with only query set
        inner_opt.zero_grad()

        with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
            if args.learn_inner_lr == True:
                for name, g in zip(fnet.param_names, diffopt.param_groups): #have to make sure both param_names and param_groups etc in same order
                    g['lr'] = fnet.inner_lrs[name]
            query_loss_array = []
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
                if args.msl == True:
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
                    query_loss_array.append(query_loss)
            
            #print("query_loss mean:", query_loss)
        if args.msl == True:
            weights = get_per_step_loss_importance_vector(it)
            query_loss = weights[0]*query_loss_array[0]
            for i in range(len(query_loss_array)-1):
                query_loss += weights[i+1]*query_loss_array[i+1]
            return query_loss
        else:
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
            return query_loss
    
    def train(it):
        model.train()
        # optimizer_global.zero_grad()
        # optimizer_local.zero_grad()
        optimizer.zero_grad()
        num_inner_loop_steps = args.inner_loop_steps #experiment with this
        batch_size_outer_loop = args.outer_batch_size #experiment with this
            
        if MAML == True:
            #MY CONTRIBUTION
            #compute inner loop each time, and then compute outer loop
            #in inner loop adapt parameters manually, then do loss.backward for outer loop
            task_batch = []
            #sample indices for batch of tasks
            tasks = np.random.choice(np.arange(0,num_tasks), args.outer_batch_size, replace=False)
            if args.curriculum == False:
                for task in tasks:
                    temp_batch = []
                    idx = np.random.choice(idx_lists[task], num_examples_per_task, replace=False) #sample index for 1 example from task subset
                    for i in idx:
                        temp_batch.append(train_set[int(i)])
                    task_batch.append(Batch.from_data_list(temp_batch).to(args.device))
            else:
                if args.curriculum_type == 2:
                    section = min(int((it-1)/(step_length*size_train_MAML)), task_split-1) #just to be safe in edge case that dataset weird size, then just sample a bit more from all dataset            
                    print(section)
                    for task in tasks:
                        temp_batch = []
                        idx = np.random.choice(idx_lists[task][0:idx_bound_tasks[section]], num_examples_per_task, replace=False) #sample index for 1 example from task subset
                        for i in idx:
                            temp_batch.append(train_set[int(i)])
                        task_batch.append(Batch.from_data_list(temp_batch).to(args.device))
                else:
                    raise NotImplementedError()
            
            # if args.curriculum_type == 0 or args.curriculum_type == 1:
            #     for _ in range(batch_size_outer_loop):
            #         task_batch.append(next(train_iterator).to(args.device))
            # elif args.curriculum_type == 2:
            #     section = int((it-1)/(step_length*size_train_MAML))
            #     print(section)
            #     idxs = np.random.choice(idx_lists[section], args.outer_batch_size, replace=False)
            #     for i in idxs:
            #         temp_batch = []
            #         for j in range(num_examples_per_task):
            #             temp_batch.append(train_set[int(i)+j].to(args.device))
            #         task_batch.append(Batch.from_data_list(temp_batch)) #okay to do +5 because idx don't get last 5 due to arange
            outer_loss_batch = []
            for task in task_batch:
                query_loss = inner_loop(model, inner_opt, task, num_inner_loop_steps, it)
                query_loss = query_loss.mean()
                query_loss.backward()
                outer_loss_batch.append(query_loss.detach())
            loss = torch.mean(torch.stack(outer_loss_batch))
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)      
            optimizer.step()
            #print(model.inner_lrs)
            
        else:
        #previous approach
            batch = []
            tasks = np.random.choice(np.arange(0,num_tasks), args.outer_batch_size, replace=False)
            if args.curriculum == False:
                for _ in range(batch_size_outer_loop):
                    batch = batch + next(train_iterator_geodiff).to(args.device).to_data_list()
            else:
                if args.curriculum_type == 0 or args.curriculum_type == 1: #also valid for no curriculum
                    raise NotImplementedError()
                elif args.curriculum_type == 2:
                    #just sample same way for consistency, in a sense similarly random
                    section = min(int((it_geodiff-1)/(step_length*size_train_geodiff)), task_split-1) #just to be safe in edge case that dataset weird size, then just sample a bit more from all dataset
                    print(section)
                    for task in tasks:
                        idx = np.random.choice(idx_lists[task][0:idx_bound_tasks[section]], num_examples_per_task, replace=False) #sample index for 1 example from task subset
                        for i in idx:
                            batch.append(train_set[int(i)].to(args.device))

            batch = Batch.from_data_list(batch)
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
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        
            optimizer.step()
            # optimizer_global.step()
            # optimizer_local.step()
        
        
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
        if MAML == True:
            #MY CONTRIBUTION
            num_inner_loop_steps = args.inner_loop_steps #experiment with this
            val_size = int(args.val_size*len(val_set))
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                if i >= val_size:
                    break
                batch = batch.to(args.device)
                loss_batch = []
                query_loss = inner_loop(model, inner_opt, batch, num_inner_loop_steps, it) #batch here is just one task
                loss_batch.append(query_loss.detach())
            avg_loss = torch.mean(torch.stack(loss_batch)) 
        else:
            for i, batch in enumerate(tqdm(val_loader, desc='Validation')):
                batch = batch.to(args.device)
                #only do validation on query
                if args.decrease_sup_size == False:
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
        init_sup_size = args.sup_size
        min_sup_size = 0
        max_iter = args.max_iter
        freq_task_change = int(size_train_MAML*epoch_size/(4)) #this is aligned with MAML/GeoDiff split
        val_freq = int(len(train_set)*args.val_freq/(args.outer_batch_size*num_examples_per_task))
        # for i in np.linspace(0,args.max_iter,10):
        #     print(get_per_step_loss_importance_vector(i))
        done = False
        for it in range(start_iter, max_iter + 1):
            if args.sup_size == 0 and done == False:
                MAML = False
                if args.curriculum_type == 1:
                    raise NotImplementedError()
                    #temp is just an original copy of train_set.data
                    train_set.data = list(np.tile(np.reshape(tr_data_copy, (epoch_split, -1)), size_train_geodiff).flatten())
                    train_iterator = inf_iterator(DataLoader(train_set, num_examples_per_task, shuffle=False))
                done = True
            if done == True and args.maml == True: #so if it got to geodiff and ~started~ with MAML true, which means doing a curriculum
                #no problem for others bc only use it_geodiff if curriculum_type = 2
                it_geodiff += 1
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
            if (args.decrease_sup_size == True) and (it % freq_task_change == 0):
                args.sup_size = max(min_sup_size, args.sup_size - 1)
    except KeyboardInterrupt:
        logger.info('Terminating...')

