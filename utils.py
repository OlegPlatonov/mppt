import os
import yaml
import torch


class Logger:
    def __init__(self, save_dir, metric, maximize_metric):
        self.save_dir = save_dir
        self.metric = metric
        self.maximize_metric = maximize_metric
        self.best_val_metric = None
        self.corresponding_test_metric = None
        self.best_epoch = None
        self.val_metric_values = []
        self.test_metric_values = []

    def update_metrics(self, val_metric, test_metric, epoch):
        self.val_metric_values.append(val_metric)
        self.test_metric_values.append(test_metric)

        if self.best_val_metric is None or (self.maximize_metric and val_metric > self.best_val_metric) or \
                (not self.maximize_metric and val_metric < self.best_val_metric):
            self.best_val_metric = val_metric
            self.corresponding_test_metric = test_metric
            self.best_epoch = epoch

        metrics = {
            f'best val {self.metric}': self.best_val_metric,
            f'corresponding test {self.metric}': self.corresponding_test_metric,
            f'best epoch': self.best_epoch,
            f'val {self.metric} values': self.val_metric_values,
            f'test {self.metric} values': self.test_metric_values
        }

        with open(os.path.join(self.save_dir, 'metrics.yaml'), 'w') as file:
            yaml.safe_dump(metrics, file, sort_keys=False)


def get_save_dir(base_dir, dataset_name, name):
    idx = 1
    save_dir = os.path.join(base_dir, dataset_name, f'{name}_{idx:02d}')
    while os.path.exists(save_dir):
        idx += 1
        save_dir = os.path.join(base_dir, dataset_name, f'{name}_{idx:02d}')

    os.makedirs(save_dir)

    return save_dir


def get_parameter_groups(model):
    no_weight_decay_names = ['bias', 'layernorm', 'input_mlp']

    parameter_groups = [
        {
            'params': [param for name, param in model.named_parameters()
                       if not any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)]
        },
        {
            'params': [param for name, param in model.named_parameters()
                       if any(no_weight_decay_name in name for no_weight_decay_name in no_weight_decay_names)],
            'weight_decay': 0
        },
    ]

    return parameter_groups


def get_lr_scheduler_with_warmup(optimizer, num_warmup_steps=None, num_steps=None, warmup_proportion=None,
                                 last_step=-1):

    if num_warmup_steps is None and (num_steps is None or warmup_proportion is None):
        raise ValueError('Either num_warmup_steps or num_steps and warmup_proportion should be provided.')

    if num_warmup_steps is None:
        num_warmup_steps = int(num_steps * warmup_proportion)

    def get_lr_multiplier(step):
        if step < num_warmup_steps:
            return (step + 1) / (num_warmup_steps + 1)
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier, last_epoch=last_step)

    return lr_scheduler
