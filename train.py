import os
import torch
import time
from tqdm import tqdm
from opts import parse_opts
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = parse_opts()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(parse_args):
    opt = parse_args

    log_path = os.path.join(opt.log_path, opt.datasetName.upper())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, time.strftime('%Y-%m-%d-%H:%M:%S' + '.log'))
    logger = ConfigLogging(log_file)
    logger.info(opt)    # 保存当前模型参数

    setup_seed(opt.seed)
    model = build_model(opt).to(device)
    model.preprocess_model(pretrain_path={
        'T': "/opt/data/private/K-MSA/pretrainedModel/SIMS_T_MAE-0.278_Corr-0.765.pth",
        'V': "/opt/data/private/K-MSA/pretrainedModel/SIMS_V_MAE-0.522_Corr-0.520.pth",
        'A': "/opt/data/private/K-MSA/pretrainedModel/SIMS_A_MAE-0.516_Corr-0.261.pth"
    })      # 加载预训练权重并冻结参数
    dataLoader = MMDataLoader(opt)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    scheduler_warmup = get_scheduler(optimizer, opt.n_epochs)

    for epoch in range(1, opt.n_epochs+1):
        train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, metrics)
        save_print_results(opt, logger, train_results, valid_results, test_results)
        scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.train()
    for data in train_pbar:
        inputs = {
            'V': data['vision'].to(device),
            'A': data['audio'].to(device),
            'T': data['text'].to(device),
            'mask': {
                'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                'T': []
            }
        }
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        copy_label = label.clone().detach()
        batchsize = inputs['V'].shape[0]

        output, nce_loss = model(inputs, copy_label)

        loss_re = loss_fn(output, label)
        loss = loss_re + 0.1 * nce_loss
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

    return train_results


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(eval_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output, _ = model(inputs, None)
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        valid_results = metrics(pred, true)

    return valid_results


def test(model, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output, _ = model(inputs, None)
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        test_results = metrics(pred, true)

    return test_results


if __name__ == '__main__':
    main(opt)

'''
| Test    | 0.4084 | 0.5985 |  0.779  |  0.6324 |  0.4245 | 0.7764 |

| Test    | 0.407  | 0.5878 |  0.7702 |  0.6499 |  0.4354 | 0.7683 |       dropout

| Test    | 0.4131 | 0.588  |  0.7856 |  0.663  |  0.4376 | 0.7839 |        # 层数还是3层，其他都没变；只改了dropout，以及添加了attention融合

| Test    | 0.4086 | 0.612  |  0.7702 |  0.6411 |  0.4289 | 0.7683 |
| Test    | 0.4082 | 0.5976 |  0.7921 |  0.6586 |  0.4442 | 0.7937 |        # rout 除了unisenti以外的东西

| Test    | 0.4047 | 0.6025 |  0.7965 |  0.6761 |  0.4486 | 0.7948 |    seed 1111 unisent SOTA

| Test    | 0.4007 | 0.6038 |  0.7921 |  0.6761 |  0.4617 | 0.7907 |    添加CPC
| Test    | 0.4075 | 0.6123 |  0.8053 |  0.6696 |  0.4464 | 0.804  |
| Test    | 0.4082 | 0.6132 |  0.8074 |  0.6652 |  0.4354 | 0.8071 |

SIMSv2
| Test    | 0.2717 | 0.7631 |  0.8269 |  0.7524 |  0.588  | 0.8263 |

| Test    | 0.2704 | 0.7624 |  0.8221 |  0.7544 |  0.5996 | 0.8213 |
'''
