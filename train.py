import os
import logging
import torch
from tqdm import tqdm
from opts import parse_opts
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, save_model, setup_seed
from models.almt import build_model
from core.metric import MetricsTop


opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print("device: {}:{}".format(device, opt.CUDA_VISIBLE_DEVICES))


def main(parse_args):
    opt = parse_args
    if opt.seed is not None:
        setup_seed(opt.seed)
    print("seed: {}".format(opt.seed))

    log_path = os.path.join(".", "log", opt.project_name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    model = build_model(opt).to(device)
    dataLoader = MMDataLoader(opt)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    scheduler_warmup = get_scheduler(optimizer, opt)
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)

    for epoch in range(1, opt.n_epochs+1):
        train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        if opt.is_test is not None:
            test(model, dataLoader['test'], optimizer, loss_fn, epoch, metrics)
        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.train()
    for data in train_pbar:
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        output = model(img, audio, text)

        loss = loss_fn(output, label)

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({
            'epoch': '{}'.format(epoch),
            'loss': '{:.5f}'.format(losses.value_avg),
            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(eval_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)


def test(model, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)

            loss = loss_fn(output, label)

            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)


if __name__ == '__main__':
    main(opt)

'''
test: {'Has0_acc_2': 0.8382, 'Has0_F1_score': 0.8402, 'Non0_acc_2': 0.8613, 'Non0_F1_score': 0.8624, 'Mult_acc_5': 0.5262, 'Mult_acc_7': 0.4606, 'MAE': 0.7066, 'Corr': 0.7899}
'''
