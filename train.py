import os
import argparse
import pandas as pd
import torch
from torch import nn
from torch import optim

from pytorchcls.dataloader import generate_dataloader, split_dataframe
from pytorchcls.training_loop import training


def parser_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe', type=str, help="Metadata path (csv file)", required=True)
    # os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'train', 'train_meta.csv')
    parser.add_argument('--img_dir', type=str, help="Directory path contain images", required=True)
    # os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', 'train', 'images')
    parser.add_argument('--saving_dir', type=str, help="Saving directory to save model checkpoint",
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', 'new'))
    # os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models')
    parser.add_argument('--model_checkpoint', type=str, help='Checkpoint of model',
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models', 'densenet_mask.pt'))
    parser.add_argument('--model_type', type=str, help='What is the purpose of model (classify distancing or mask)',
                        default=r'mask')
    parser.add_argument('--height', type=int, help='Height Image', default=300)
    parser.add_argument('--width', type=int, help='Width Image', default=300)
    parser.add_argument('--batch', type=int, help="Batch size", default=8)
    parser.add_argument('--epochs', type=int, help="Number epoch", default=1)

    return parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser_argument()
    if not os.path.exists(args.saving_dir):
        os.makedirs(args.saving_dir)

    df = pd.read_csv(args.dataframe, usecols=['fname', args.model_type], nrows=10)
    dfs = split_dataframe(df, dataset_size=(0.9, 0.1, 0))
    dataloader = generate_dataloader(dfs, [args.img_dir] * 3, batch_size=args.batch)

    model = torch.load(args.model_checkpoint, map_location=device)

    loss_func = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0005)
    model = training(model=model, optimizer=optimizer, loss_func=loss_func,
                     dataloader=dataloader, device=device,
                     saving_dir=args.saving_dir, num_epochs=args.epochs)
