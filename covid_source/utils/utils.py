import logging
from tqdm import tqdm
import numpy as np
import random
import os
import torch
from sklearn import metrics
from utils.dataloader import make_trainloader, make_testloader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStop:
    def __init__(self, patience=10, mode='max', delta=0.0001):
        self.patientce = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == 'min':
            # NumPy 2.0 removed the deprecated ``np.Inf`` alias.
            self.val_score = np.inf
        else:
            # NumPy 2.0 removed the deprecated ``np.Inf`` alias.
            self.val_score = -np.inf

    def __call__(self, epoch_score, model, model_path):
        if self.mode == 'min':
            score = -1. * epoch_score
        else:
            score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patientce:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score


class Engine:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None,
                 criterion=None, optimizer=None, scheduler=None,
                 model=None, device='cpu'):

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.model = model
        self.device = device

    def train_epoch(self, epoch, binary=False):

        losses = AverageMeter()
        self.model.train()

        bar = tqdm(self.train_loader)
        for _, (data, target) in enumerate(bar):

            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            if binary:
                output = torch.sigmoid(output)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), target.size(0))
            bar.set_description(
                f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')

        logging.info(f'Epoch{epoch:3d}, train loss:{losses.avg:6f}')
        return losses.avg

    def evaluate_epoch(self, epoch, binary=False):

        accuracies = AverageMeter()
        losses = AverageMeter()
        flag = False
        self.model.eval()
        bar = tqdm(self.val_loader)
        with torch.no_grad():
            for _, (data, target) in enumerate(bar):

                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                if binary:
                    output = torch.sigmoid(output)
                loss = self.criterion(output, target)
                losses.update(loss.item(), target.size(0))

                target = target.detach().cpu().numpy()

                if target.ndim == 1:
                    flag = True
                    prediction = torch.argmax(output, dim=1).view(-1).detach().cpu().numpy()
                    accuracy = (prediction == target).mean()
                    accuracies.update(float(accuracy), data.size(0))
                    bar.set_description(
                        f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , accuracy:{accuracies.avg:6f}')
                elif binary:
                    flag = True
                    prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    accuracy = metrics.accuracy_score(target, prediction)
                    accuracies.update(float(accuracy), data.size(0))
                    bar.set_description(
                        f'Epoch{epoch:3d}, valid loss:{losses.avg:6f} , valid accuracy:{accuracies.avg:6f}')
                else:
                    bar.set_description(
                        f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')

        if self.scheduler:
            self.scheduler.step()
        if target.ndim == 1 or binary:
            logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}, valid accuracy:{accuracies.avg:6f}')
        else:
            logging.info(f'Epoch{epoch:3d}, valid loss:{losses.avg:6f}')

        return losses.avg, accuracies.avg if flag else None

    def test_epoch(self, binary=False):

        accuracies = AverageMeter()
        losses = AverageMeter()

        self.model.eval()
        bar = tqdm(self.test_loader)
        outputs = []
        predicted = []
        true = []
        with torch.no_grad():
            for _, (data, target) in enumerate(bar):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                if binary:
                    output = torch.sigmoid(output)

                loss = self.criterion(output, target)

                target = target.detach().cpu().numpy()
                if binary:
                    prediction = torch.greater_equal(output, 0.5).to(torch.float64).cpu().detach().numpy()
                    accuracy = metrics.accuracy_score(target, prediction)

                else:
                    prediction = torch.argmax(output, dim=1).view(-1).detach().cpu().numpy()
                    accuracy = (prediction == target).mean()

                outputs.append(output.detach().cpu().numpy())

                losses.update(loss.item(), data.size(0))
                accuracies.update(float(accuracy), data.size(0))
                bar.set_description(
                    f"test loss: {losses.avg:.5f} accuracy:{accuracies.avg:.5f}")
                predicted += prediction.tolist()
                true += target.tolist()
            outputs = np.concatenate(outputs)
        return outputs, np.array(predicted), np.array(true), accuracies.avg, losses.avg


def load_net_state(net, state_dict, allow_unused_wave_emb=False):
    if allow_unused_wave_emb:
        legacy_keys = {'wave_emb.0.weight', 'wave_emb.0.bias', 'wave_emb.1.weight',
                       'wave_emb.1.bias', 'wave_emb.1.running_mean',
                       'wave_emb.1.running_var', 'wave_emb.1.num_batches_tracked'}
        removed = sorted(set(state_dict) & legacy_keys)
        if removed:
            logging.warning('Explicitly discarding legacy keys absent from published model: %s', removed)
        state_dict = {k: v for k, v in state_dict.items() if k not in legacy_keys}
    net.load_state_dict(state_dict, strict=True)
    return net


def train_model(model, save_path, ds, device='cpu', tune=False,  **kwargs):
    from torch.optim.lr_scheduler import CosineAnnealingLR

    pool_dim = 256 if model.__class__.__name__ == 'Pace' else 1024

    train_loader, val_loader = make_trainloader(
        ds, batch_size=kwargs['batch_size'],
        num_workers=12, train_size=kwargs['train_size'],
        seed=42, tune=tune, pool_dim=pool_dim, resize_dim=kwargs.get('linear_resize'))

    binary = True if ds == 'FunctionalGroups' else False
    criterion = torch.nn.BCELoss() if binary else torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), **kwargs['Adam_params'])

    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    logging.info('Effective optimizer: %s, defaults=%s', type(optimizer).__name__, optimizer.defaults)
    logging.info('Effective scheduler: %s, T_max=%s, eta_min=%s, step=once_per_validation_epoch',
                 type(scheduler).__name__, scheduler.T_max, scheduler.eta_min)
    logging.info('Effective split: train=%s, valid=%s, split_seed=42, train_size=%s',
                 len(train_loader.dataset), len(val_loader.dataset), kwargs['train_size'])

    # mode = 'min' if not tune else 'max'
    mode = 'max'
    es = EarlyStop(patience=kwargs['patience'], mode=mode)

    engine = Engine(train_loader=train_loader, val_loader=val_loader,
                    criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                    model=model, device=device)

    # start to train 
    for epoch in range(kwargs['epoch']):
        train_loss = engine.train_epoch(epoch, binary=binary)
        val_loss, acc = engine.evaluate_epoch(epoch, binary=binary)
        # Select only with held-out validation labels. In particular, 0.0 is
        # a valid accuracy and must never fall back to maximizing val_loss.
        if acc is not None:
            es(acc, model, f'{save_path}.pth')
        else:
            raise ValueError('Validation accuracy unavailable for this classification task')
        if es.early_stop:
            break
    print(es.val_score)


def test_model(model, ds, device='cpu', verbose=True, linear_resize=None):
    pool_dim = 1024
    test_loader = make_testloader(ds, pool_dim=pool_dim, resize_dim=linear_resize)
    binary = True if ds == 'FunctionalGroups' else False
    criterion = torch.nn.BCELoss() if binary else torch.nn.CrossEntropyLoss()
    engine = Engine(test_loader=test_loader,
                    criterion=criterion, model=model, device=device)
    outputs, pred, true, _, _ = engine.test_epoch(binary=binary)

    if verbose:
        from sklearn import metrics
        print(metrics.classification_report(true, pred, digits=4))
        logging.info(metrics.classification_report(true, pred, digits=4))
        logging.info(f'accuracy:{metrics.accuracy_score(true, pred):.5f}')

    return outputs, pred, true
