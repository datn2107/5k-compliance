import os
import time
import numpy as np
import torch
import argparse
import pandas as pd

from torchvision import io
from torchvision.transforms import Compose, Resize, Normalize
from pytorchcls.model import EnsembleMultiModel


def load_model(saved_models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    mask_model_ckpt = os.path.join(saved_models, 'densenet_mask.pt')
    mask_model = torch.load(mask_model_ckpt, map_location=device)

    dis_model_ckpts = [os.path.join(saved_models, 'densenet_dis.pt'),
                       os.path.join(saved_models, 'regnet_dis.pt')]
    dis_models = [torch.load(dis_model_ckpt, map_location=device) for dis_model_ckpt in dis_model_ckpts]
    dis_model = EnsembleMultiModel(dis_models)

    return mask_model, dis_model


def predict(df, img_dir, model, batch_size=8, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imgs = []
    for index, row in df.iterrows():
        fname = row['fname']
        img_path = os.path.join(img_dir, fname)

        transforms = Compose([
            Resize((300, 300)),
            Normalize(mean=(0, 0, 0), std=(255, 255, 255)),
        ])
        imgs.append(transforms(io.read_image(img_path).float()))
    img_batches = torch.split(torch.stack(imgs), batch_size)

    result = []
    model = model.to(device)
    for img_batch in img_batches:
        img_batch = img_batch.to(device)

        output = torch.gt(model(img_batch), threshold)
        # output will have shape (batch, 1) but we expect it is (batch) so we add .squeeze
        output = output.squeeze().detach().cpu().numpy()
        result.extend(output)

    return result


def parser_argument():
    source_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help="Directory contain images_with_box and .csv file", required=True)
    parser.add_argument('--saving_dir', type=str, help="Saving directory to save model checkpoint",
                        default=os.path.join(os.path.dirname(source_dir), 'result'))
    # os.path.join(os.path.dirname(os.path.realpath(__file__)), 'saved_models')
    parser.add_argument('--saved_models', type=str, help='Directory that contain models checkpoint',
                        default=os.path.join(source_dir, 'saved_models'))
    parser.add_argument('--height', type=int, help='Height Image', default=300)
    parser.add_argument('--width', type=int, help='Width Image', default=300)
    parser.add_argument('--batch', type=int, help="Batch size", default=8)
    parser.add_argument('--mask_threshold', type=float, help='Threshold for mask model prediction', default=0.5)
    parser.add_argument('--dis_threshold', type=float, help='Threshold for distance model prediction', default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = parser_argument()
    if not os.path.exists(args.saving_dir):
        os.makedirs(args.saving_dir)
    img_dir = os.path.join(args.data, "images_with_box")
    csv_file = [file for file in os.listdir(args.data) if file.endswith('.csv')][0]
    csv_path = os.path.join(args.data, csv_file)

    mask_model, dis_model = load_model(args.saved_models)
    df = pd.read_csv(csv_path)

    start_time = time.time()
    mask_result = predict(df, img_dir, model=mask_model, batch_size=args.batch, threshold=args.mask_threshold)
    dis_result = predict(df, img_dir, model=dis_model, batch_size=args.batch, threshold=args.dis_threshold)
    end_time = time.time()

    result = np.logical_and(mask_result, dis_result).astype(np.float32)
    df['5K'] = result

    df.to_csv(os.path.join(args.saving_dir, 'submission.csv'), index=False)

    total_time = end_time - start_time
    print("Inference time: {:.0f}m{:.0f}s".format(total_time / 60, total_time % 60))
