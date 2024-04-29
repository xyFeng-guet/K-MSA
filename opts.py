import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',
                 type=str,
                 default='sims',
                 help='mosi, mosei or sims'),
            dict(name='--dataPath',
                 default="/opt/data/private/Project/Datasets/MSA_Datasets/SIMS/Processed/unaligned_39.pkl",
                 type=str,
                 help=' '),
            dict(name='--seq_lens',
                 default=[50, 50, 50],
                 type=list,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
            dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
        ],

        'network': [
            dict(name='--fusion_layer_depth',
                 default=2,
                 type=int)
        ],

        'common': [
            dict(name='--seed',  # try different seeds
                 default=1111,
                 type=int),
            dict(name='--batch_size',
                 default=64,
                 type=int,
                 help=' '),
            dict(name='--lr',
                 type=float,
                 default=1e-4),
            dict(name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(name='--n_epochs',
                 default=200,
                 type=int,
                 help='Number of total epochs to run'),
            dict(name='--log_path',
                 default='./log',
                 type=str,
                 help='the logger path for save options and experience results')
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args
