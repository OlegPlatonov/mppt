import os
import argparse
import yaml
from tqdm import tqdm

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from ema_pytorch import EMA

from ogb.graphproppred import Evaluator

from model import Model
from dataset import MoleculeDataset
from utils import Logger, get_save_dir, get_parameter_groups, get_lr_scheduler_with_warmup
from mappings import metric_names, ogb_metric_names, maximize_metric


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Experiment name.')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Base directory for saving information.')
    parser.add_argument('--dataset', type=str, required=True)

    # model architecture
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--hidden_dim_multiplier', type=float, default=4)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--save_model', default=False, action='store_true')

    # regularization
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attn_dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Only used if num_warmup_steps is None.')

    parser.add_argument('--roc_auc', default=False, action='store_true',
                        help='Only used for ogbg-molpcba and ogbg-molmuv datasets. Compute ROC AUC instead of AP.')

    # EMA parameters
    parser.add_argument('--ema_update_after_step', type=int, default=1e12,
                        help='Set to longer than training to disable EMA.')
    parser.add_argument('--ema_beta', type=float, default=0.999)
    parser.add_argument('--ema_update_every', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)

    args = parser.parse_args()

    return args


def compute_loss(preds, y, dataset_name):
    if dataset_name in ('ogbg-molpcba', 'ogbg-molmuv', 'ogbg-moltoxcast', 'ogbg-moltox21'):
        labeled_mask = ~y.isnan()
        loss = F.binary_cross_entropy_with_logits(input=preds[labeled_mask], target=y[labeled_mask], reduction='mean')
    elif dataset_name in ('ogbg-molhiv', 'ogbg-molbbbp', 'ogbg-molbace', 'ogbg-molclintox', 'ogbg-molsider'):
        loss = F.binary_cross_entropy_with_logits(input=preds, target=y, reduction='mean')
    elif dataset_name in ('ogbg-mollipo', 'ogbg-molesol', 'ogbg-molfreesolv'):
        loss = F.mse_loss(input=preds, target=y)
    elif dataset_name == 'pcqm4mv2':
        loss = F.l1_loss(input=preds, target=y, reduction='mean')
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}.')

    return loss


def train_epoch(model, data_loader, optimizer, scaler, scheduler, ema, tb_writer, dataset_name, epoch, step,
                num_processed_samples, device, amp=False, num_accumulation_steps=1):

    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_samples = len(data_loader) * data_loader.batch_size
    with tqdm(total=num_samples, desc=f'Epoch {epoch}') as progress_bar:
        for i, (x, rels, attn_mask, y) in enumerate(data_loader, start=1):
            x, rels, attn_mask, y = x.to(device), rels.to(device), attn_mask.to(device), y.to(device)

            with autocast(enabled=amp):
                preds = model(x, rels, attn_mask)
                loss = compute_loss(preds=preds, y=y, dataset_name=dataset_name) / num_accumulation_steps

            scaler.scale(loss).backward()

            loss_value = loss.item() * num_accumulation_steps
            cur_lr = scheduler.get_last_lr()[0]
            cur_batch_size = len(x)
            num_processed_samples += cur_batch_size
            tb_writer.add_scalar(tag='train loss', scalar_value=loss_value, global_step=num_processed_samples)
            progress_bar.update(cur_batch_size)
            progress_bar.set_postfix(step=step, lr=f'{cur_lr:.2e}', loss=f'{loss_value:.4f}')

            if i % num_accumulation_steps == 0:
                tb_writer.add_scalar(tag='lr', scalar_value=cur_lr, global_step=num_processed_samples)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update()
                step += 1

    return step, num_processed_samples


@torch.no_grad()
def compute_metric(preds, y, dataset_name, roc_auc=False):
    if dataset_name == 'pcqm4mv2':
        preds = preds.clip(min=0, max=50)
        metric = np.mean(np.abs(preds - y))
    else:
        evaluator = Evaluator(name=dataset_name)

        if dataset_name in ('ogbg-molpcba', 'ogbg-molmuv') and roc_auc:
            evaluator.eval_metric = 'rocauc'

        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
            y = y.reshape(-1, 1)

        metric = evaluator.eval({'y_pred': preds, 'y_true': y})[ogb_metric_names[dataset_name]]

    return metric


@torch.no_grad()
def evaluate(model, data_loader, tb_writer, dataset_name, split, num_processed_samples, device, amp=False,
             roc_auc=False):

    model.eval()
    all_preds = []
    all_targets = []
    for x, rels, attn_mask, y in data_loader:
        x, rels, attn_mask = x.to(device), rels.to(device), attn_mask.to(device)

        with autocast(enabled=amp):
            preds = model(x, rels, attn_mask)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metric = compute_metric(preds=all_preds, y=all_targets, dataset_name=dataset_name, roc_auc=roc_auc)

    split = split[0].upper() + split[1:]
    metric_name = metric_names[dataset_name]
    print(f'{split} {metric_name}: {metric:.4f}')
    tb_writer.add_scalar(tag=f'{split.lower()} {metric_name}', scalar_value=metric, global_step=num_processed_samples)

    return metric


def main():
    args = get_args()

    dataset_name = args.dataset.split('_')[0]

    save_dir = get_save_dir(base_dir=args.save_dir, dataset_name=dataset_name, name=args.name)
    print(f'Results will be saved to {save_dir}.')

    with open(os.path.join(save_dir, 'args.yaml'), 'w') as file:
        yaml.safe_dump(vars(args), file, sort_keys=False)

    tb_writer = SummaryWriter(log_dir=save_dir)

    if dataset_name in ('ogbg-molpcba', 'ogbg-molmuv') and args.roc_auc:
        metric_names[dataset_name] = 'ROC AUC'
        ogb_metric_names[dataset_name] = 'rocauc'

    logger = Logger(save_dir=save_dir, metric=metric_names[dataset_name], maximize_metric=maximize_metric[dataset_name])

    print('Preparing data...')
    train_dataset = MoleculeDataset(name=args.dataset, split='train')
    val_dataset = MoleculeDataset(name=args.dataset, split='val')
    test_dataset = MoleculeDataset(name=args.dataset, split='test')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=train_dataset.collate_fn,
                              shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=val_dataset.collate_fn,
                            shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn,
                             shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    num_steps = len(train_loader) * args.num_epochs // args.num_accumulation_steps

    print('Creating model...')
    model = Model(input_dim=train_dataset.input_dim,
                  num_targets=train_dataset.num_targets,
                  num_layers=args.num_layers,
                  hidden_dim=args.hidden_dim,
                  num_heads=args.num_heads,
                  hidden_dim_multiplier=args.hidden_dim_multiplier,
                  num_rels=train_dataset.num_rels,
                  dropout=args.dropout,
                  attn_dropout=args.attn_dropout)

    if args.pretrained_model is not None:
        state_dict = torch.load(args.pretrained_model)

        del state_dict['output_linear.weight']
        del state_dict['output_linear.bias']

        model.load_state_dict(state_dict, strict=False)

    model.to(args.device)

    ema = EMA(model, beta=args.ema_beta, update_after_step=args.ema_update_after_step,
              update_every=args.ema_update_every)

    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    scheduler = get_lr_scheduler_with_warmup(optimizer=optimizer, num_steps=num_steps,
                                             num_warmup_steps=args.num_warmup_steps,
                                             warmup_proportion=args.warmup_proportion)

    print('Starting training...')
    best_val_metric = None
    corresponding_test_metric = None
    step = 1
    num_processed_samples = 0
    for epoch in range(1, args.num_epochs + 1):
        step, num_processed_samples = train_epoch(model=model, data_loader=train_loader, optimizer=optimizer,
                                                  scaler=scaler, scheduler=scheduler, ema=ema, tb_writer=tb_writer,
                                                  dataset_name=dataset_name, epoch=epoch, step=step,
                                                  num_processed_samples=num_processed_samples, device=args.device,
                                                  amp=args.amp, num_accumulation_steps=args.num_accumulation_steps)

        print('Evaluating...')
        val_metric = evaluate(model=ema, data_loader=val_loader, tb_writer=tb_writer, dataset_name=dataset_name,
                              split='val', num_processed_samples=num_processed_samples, device=args.device,
                              amp=args.amp, roc_auc=args.roc_auc)

        if dataset_name != 'pcqm4mv2':
            test_metric = evaluate(model=ema, data_loader=test_loader, tb_writer=tb_writer, dataset_name=dataset_name,
                                   split='test', num_processed_samples=num_processed_samples, device=args.device,
                                   amp=args.amp, roc_auc=args.roc_auc)
        else:
            test_metric = None

        print()

        logger.update_metrics(val_metric=val_metric.item(), test_metric=test_metric.item(), epoch=epoch)

        if best_val_metric is None or (maximize_metric[dataset_name] and val_metric > best_val_metric) or \
                (not maximize_metric[dataset_name] and val_metric < best_val_metric):
            best_val_metric = val_metric
            corresponding_test_metric = test_metric
            if args.save_model:
                torch.save(ema.ema_model.state_dict(), os.path.join(save_dir, 'model.pt'))

    metric_name = metric_names[dataset_name]
    print(f'Best val {metric_name}: {best_val_metric:.4f}')
    if dataset_name != 'pcqm4mv2':
        print(f'Corresponding test {metric_name}: {corresponding_test_metric:.4f}')

    print()


if __name__ == '__main__':
    main()
