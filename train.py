import os
import argparse
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model import MPPT
from dataset import MoleculeDataset, collate_fn
from utils import get_save_dir, get_parameter_groups, get_lr_scheduler


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

    # regularization
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--attn_dropout', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_warmup_steps', type=int, default=None,
                        help='If None, warmup_proportion is used instead.')
    parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Only used if num_warmup_steps is None.')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--amp', default=False, action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    return args


def train_epoch(model, data_loader, optimizer, scaler, scheduler, epoch, device, amp=False):
    model.train()
    with tqdm(total=len(data_loader) * data_loader.batch_size, desc=f'Epoch {epoch}') as progress_bar:
        for x, attn_mask, y in data_loader:
            x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)

            with autocast(enabled=amp):
                preds = model(x, attn_mask)
                loss = F.l1_loss(input=preds, target=y, reduction='mean')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            progress_bar.update(len(x))
            progress_bar.set_postfix(loss=f'{loss.item():.4f}')


@torch.no_grad()
def evaluate(model, data_loader, device, amp=False):
    print('Evaluating...')
    model.eval()
    total_loss = 0
    for x, attn_mask, y in data_loader:
        x, attn_mask, y = x.to(device), attn_mask.to(device), y.to(device)

        with autocast(enabled=amp):
            preds = model(x, attn_mask)
            preds = preds.clamp(min=0, max=50)
            loss = F.l1_loss(input=preds, target=y, reduction='sum')

        total_loss += loss.item()

    mean_loss = total_loss / len(data_loader.dataset)
    print(f'Validation loss: {mean_loss:.4f}\n')

    return mean_loss


def main():
    args = get_args()

    save_dir = get_save_dir(base_dir=args.save_dir, name=args.name)
    print(f'Results will be saved to {save_dir}.')

    print('Preparing data...')
    train_dataset = MoleculeDataset(name=args.dataset, split='train')
    val_dataset = MoleculeDataset(name=args.dataset, split='val')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                              shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                            shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=True)

    num_steps = len(train_loader) * args.num_epochs

    print('Creating model...')
    model = MPPT(input_dim=train_dataset.input_dim,
                 num_targets=train_dataset.num_targets,
                 num_layers=args.num_layers,
                 hidden_dim=args.hidden_dim,
                 num_heads=args.num_heads,
                 hidden_dim_multiplier=args.hidden_dim_multiplier,
                 dropout=args.dropout,
                 attn_dropout=args.attn_dropout)
    model.to(args.device)

    parameter_groups = get_parameter_groups(model)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=args.amp)
    scheduler = get_lr_scheduler(optimizer=optimizer, num_steps=num_steps, num_warmup_steps=args.num_warmup_steps,
                                 warmup_proportion=args.warmup_proportion)

    print('Starting training...')
    best_val_loss = None
    for epoch in range(1, args.num_epochs + 1):
        train_epoch(model=model, data_loader=train_loader, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
                    epoch=epoch, device=args.device, amp=args.amp)

        val_loss = evaluate(model=model, data_loader=val_loader, device=args.device, amp=args.amp)

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))


if __name__ == '__main__':
    main()
