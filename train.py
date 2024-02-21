import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                if args.model == 'bert':
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def test(model, criterion, dataset, logger, args, show_progress=False):
    model.eval()

    loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['test_data'],
        step_size=args.robust_step_size,
        alpha=args.alpha)

    loader = dataset['test_loader']
    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    all_predictions = []
    all_aux_labels = []
    all_labels = []

    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            if args.model == 'bert':
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y
                )[1] # [1] returns logits
            else:
                outputs = model(x)

            all_predictions.append(torch.argmax(outputs, 1).detach().cpu())
            all_aux_labels.append(g.detach().cpu())
            all_labels.append(y.detach().cpu())
            
            loss_main = loss_computer.loss(outputs, y, g, False)

        loss_computer.log_stats(logger, False)

    all_predictions = torch.cat(all_predictions)
    all_aux_labels = torch.cat(all_aux_labels)
    all_labels = torch.cat(all_labels)

    grad_norm, loss = feat_norm(model, prog_bar_loader, criterion, if_grad=True, flatten=False)
    # feat_norm = feat_norm(model, prog_bar_loader, criterion, if_grad=False, flatten=False)
    
    d = {}

    d['labels'] = all_labels.cpu()
    d['predictions'] = all_predictions.cpu()
    d['aux_labels'] = all_aux_labels.cpu()

    d['loss'] = loss
    d['grad_norm'] = grad_norm.cpu()
    # d['feat_norm'] = feat_norm.cpu()
    df = pd.DataFrame(d)
    df.to_csv('output_test_grad_loss.csv', index=True)


def feat_norm(model, data_loader, criterion, feat_type='top', flatten=False, if_grad=False, choose_gradients="last_block"):
    norm_list = []
    loss_list = []
    for test_data in data_loader:
        test_inputs, test_targets = test_data[0].cuda(), test_data[1].cuda()
        if if_grad:
            test_features, loss_test = get_grad_loss(model, test_inputs, test_targets, criterion, choose_gradients=choose_gradients, flatten=flatten)
            norm = torch.norm(F.relu(test_features), dim=[3,4]).mean(1).mean(1)  # [128, 64, 3, 7, 7]
        else:
            test_features = get_feature(model, test_inputs, feat_type=feat_type, flatten=flatten)
            norm = torch.norm(F.relu(test_features), dim=[2,3]).mean(1)  # [128, 2048, 7, 7]
        
        # print(norm.shape)
        norm_list.append(norm.data)

        if if_grad:
            loss_list += loss_test

    if if_grad:
        return torch.cat(norm_list, dim=0), loss_list
    return torch.cat(norm_list, dim=0)


def get_grad_loss(model, test_data, targets, loss_function, choose_gradients="last_block", flatten=False):
    grads_list = []
    loss_list = []
    for i, (test_data_each, targets_each) in tqdm(enumerate(zip(test_data, targets)), leave=False):
        test_data_each = test_data_each.unsqueeze(0)
        targets_each = targets_each.unsqueeze(0)
        predictions = model(test_data_each)
        loss = loss_function(predictions, targets_each)
        loss_list.append(loss.item())
        if choose_gradients == "last_block":
            block = nn.Sequential(*list(model.children())[-1:])  # TODO
            grads = torch.autograd.grad(loss, [param for param in block.parameters()])[0]
        elif choose_gradients == "all":
            grads = torch.autograd.grad(loss, [param for param in model.parameters()])[0]
        
        if flatten:
            grads_list.append(grads.view(1, -1))
        else:
            grads_list.append(grads.unsqueeze(0))  # [64, 3, 7, 7] -> [1, 64, 3, 7, 7]
    
    grads_list = torch.cat(grads_list, dim=0)
    return grads_list, loss_list


def get_feature(model, x, feat_type='top', flatten=True):
    if feat_type == 'x':
        if flatten == False:
            return x
        return x.view(x.size(0), -1)
    elif feat_type == 'top':
        feature_layers = nn.Sequential(*list(model.children())[:-1])  # TODO
        feat = feature_layers(x)
        if flatten == False:
            return feat
        return feat.view(feat.size(0), -1)
    else:
        sys.exit(1)


def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    # BERT uses its own scheduler and optimizer
    if args.model == 'bert':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)
        t_total = len(dataset['train_loader']) * args.n_epochs
        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay)
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)
        else:
            scheduler = None

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha)
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                None, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')
