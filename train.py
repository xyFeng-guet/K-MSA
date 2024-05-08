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
        'T': "/opt/data/private/K-MSA/pretrainedModel/SIMS_T_MAE-0.287_Corr-0.747.pth",
        'V': "/opt/data/private/K-MSA/pretrainedModel/SIMS_V_MAE-0.509_Corr-0.548.pth",
        'A': "/opt/data/private/K-MSA/pretrainedModel/SIMS_A_MAE-0.562_Corr-0.182.pth"
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
                'V': data['vision_padding_mask'][:, 0:data['vision'].shape[1]+1].to(device),
                'A': data['audio_padding_mask'][:, 0:data['audio'].shape[1]+1].to(device),
                'T': []
            }
        }
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = inputs['V'].shape[0]

        output = model(inputs)

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
                    'V': data['vision_padding_mask'][:, 0:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 0:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output = model(inputs)
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
                    'V': data['vision_padding_mask'][:, 0:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 0:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            output = model(inputs)
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
test: {'Has0_acc_2': 0.8382, 'Has0_F1_score': 0.8402, 'Non0_acc_2': 0.8613, 'Non0_F1_score': 0.8624, 'Mult_acc_5': 0.5262, 'Mult_acc_7': 0.4606, 'MAE': 0.7066, 'Corr': 0.7899}
| Test    | 0.4078 | 0.608  |  0.7746 |  0.6718 |  0.4136 | 0.7697 |
| Test    | 0.407  | 0.5878 |  0.7702 |  0.6499 |  0.4354 | 0.7683 |
| Test    | 0.4132 | 0.5782 |  0.7746 |  0.663  |  0.453  | 0.7723 |
| Test    | 0.4085 | 0.594  |  0.7746 |  0.6499 |  0.4464 | 0.7727 |
| Test    | 0.4272 | 0.5941 |  0.8053 |  0.6368 |  0.4026 | 0.8134 |       提高参数量
| Test    | 0.4084 | 0.5985 |  0.779  |  0.6324 |  0.4245 | 0.7764 |


| Test    | 0.407  | 0.5878 |  0.7702 |  0.6499 |  0.4354 | 0.7683 |       dropout
| Test    | 0.4132 | 0.5782 |  0.7746 |  0.663  |  0.453  | 0.7723 |

提高参数,提高dropout

| Test    | 0.4144 | 0.5833 |  0.7681 |  0.6608 |  0.442  | 0.7651 |
| Test    | 0.4121 | 0.5932 |  0.7768 |  0.6521 |  0.4333 | 0.7727 |

| Test    | 0.4125 | 0.5916 |  0.7965 |  0.6761 |  0.4158 | 0.7976 |
| Test    | 0.4131 | 0.588  |  0.7856 |  0.663  |  0.4376 | 0.7839 |        # 层数还是3层，其他都没变；只改了dropout，以及添加了attention融合

| Test    | 0.4186 | 0.571  |  0.7702 |  0.6368 |  0.4223 | 0.77   |    rout


| Test    | 0.4086 | 0.612  |  0.7702 |  0.6411 |  0.4289 | 0.7683 |
| Test    | 0.4082 | 0.5976 |  0.7921 |  0.6586 |  0.4442 | 0.7937 |
'''
