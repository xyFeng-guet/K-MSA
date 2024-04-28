import argparse
import torch
from tqdm import tqdm
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, save_model, save_print_results
from models.Encoder_KIAdapter import build_pretrained_model
from core.metric import MetricsTop


def parse_opts():
    parser = argparse.ArgumentParser(description='Pretrained Adapter')

    parser.add_argument('--datasetName', type=str, default='external_knowledge',
                        help='select external knowledge base for pre-training')
    parser.add_argument('--train_mode', type=str, default='regression',
                        help='type of pre-training labels')

    parser.add_argument('--dataPath', type=str, default='/opt/data/private/Project/Datasets/MSA_Datasets/SIMS/Processed/unaligned_39.pkl',
                        help='path for checkpointing')
    parser.add_argument('--savePath', type=str, default='./pretrainedModel/',
                        help='path for checkpointing')

    parser.add_argument('--seed', type=int, default=111,
                        help='random seed')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='epoch number for training')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='')
    parser.add_argument('--seq_lens', type=list, default=[50, 50, 200],
                        help='features length of each modality for pre-training')
    parser.add_argument('--lr', type=int, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=int, default=1e-4,
                        help='learning rate')

    args = parser.parse_args()
    return args


def train(modality, model, device, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.train()
    for data in train_pbar:
        img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
        label = data['labels'][modality].to(device)
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

    return train_results


def evaluate(modality, model, device, eval_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(eval_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels'][modality].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)
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


def test(modality, model, device, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            img, audio, text = data['vision'].to(device), data['audio'].to(device), data['text'].to(device)
            label = data['labels'][modality].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            output = model(img, audio, text)
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


def main(modality):
    opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup_seed(opt.seed)

    model = build_pretrained_model(modality).to(device)
    dataLoader = MMDataLoader(opt)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    scheduler_warmup = get_scheduler(optimizer, opt)

    for epoch in range(1, opt.n_epochs+1):
        train_results = train(modality, model, device, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(modality, model, device, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(modality, model, device, dataLoader['test'], optimizer, loss_fn, epoch, metrics)

        save_print_results(opt, None, train_results, valid_results, test_results)
        scheduler_warmup.step()

    # 保存单模态预训练模型
    save_model(opt.savePath, test_results, modality, model)


if __name__ == '__main__':
    for m in ["V"]:     # , "T", "A"
        main(modality=m)
