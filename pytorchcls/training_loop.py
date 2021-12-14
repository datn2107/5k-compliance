import os
import copy
import time
import numpy as np
import pandas as pd
from typing import List

import torch
import torchvision
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm


def init_log_loss(last_log_loss_csv):
    last_best_loss = np.inf
    log_loss = {}
    if last_log_loss_csv is None:
        log_loss['Epoch'] = []
        log_loss['train_loss'] = []
        log_loss['val_loss'] = []
    else:
        last_log_loss_df = pd.read_csv(last_log_loss_csv)
        last_log_loss = {key: list(last_log_loss_df[key]) for key in last_log_loss_df.columns}

        last_best_loss = min(last_log_loss['val_loss'])
        log_loss = last_log_loss

    return log_loss, last_best_loss


def training(model, optimizer, loss_func, dataloader, device, saving_dir,
             batch_scheduler=None, epoch_scheduler=None, num_epochs=5,
             last_epoch=0, last_log_loss_csv: str = None):
    # Model Init
    model = model.to(device)

    # Load the log_loss and also the last_best_avg_loss_all
    # if there last_log_loss_csv not exist best_avg_loss_all = np.inf
    log_loss, best_loss = init_log_loss(last_log_loss_csv)

    # Start epoch
    for epoch in range(last_epoch + 1, num_epochs + 1):
        print('-' * 50)
        print("Epoch {}/{}".format(epoch, num_epochs))

        # Init essential variable
        log_loss['Epoch'].append(epoch)

        for phase in ['train', 'val']:
            num_sample = torch.tensor(dataloader[phase].dataset.__len__(), device=device)
            with tqdm(dataloader[phase], desc="Epoch {} {}".format(epoch, phase), unit="batch") as tepoch:
                # Init sum loss (to compute average sum)
                sum_loss = torch.tensor(0.0, device=device)
                for images, targets in tepoch:
                    images = images.to(device)
                    targets = targets.to(device)
                    optimizer.zero_grad()

                    # Predict and Compute loss
                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(images)
                        loss = loss_func(preds, targets)
                        if phase == 'train':
                            loss.backward()
                    if phase == 'train':
                        optimizer.step()
                        if batch_scheduler is not None:
                            batch_scheduler.step()

                    if loss_func.reduction == 'avg':
                        sum_loss += loss * images.size(dim=0)
                    else:
                        sum_loss += loss
                    tepoch.set_postfix(loss=loss.item())
                # Add loss to log loss
                log_loss[phase + '_loss'].append((sum_loss / num_sample).item())

        if epoch_scheduler is not None:
            epoch_scheduler.step()

        # Save model and log_loss file
        if log_loss['val_loss'][-1] < best_loss:
            best_loss = log_loss['val_loss'][-1]
            # Save the best model
            torch.save(model, os.path.join(saving_dir, 'best_model.pt'))
            # Save models
            torch.save(model, os.path.join(saving_dir, 'model_epoch_{}.pt'.format(epoch)))
            # Save the log_loss
            pd.DataFrame(log_loss).to_csv(os.path.join(saving_dir, 'log_loss.csv'), index=False)

    print('Best Val Acc: {:.4f}'.format(best_loss))

    model = torch.load(os.path.join(saving_dir, 'best_model.pt'))
    return model


if __name__ == '__main__':
    pass
    # model = torch.load(r'D:\Machine Learning Project\5kCompliance\5kCompliance\best_model_0.pt',
    #                    map_location=torch.device('cpu'))
    # torch.save(model.state_dict(), r'D:\Machine Learning Project\5kCompliance\5kCompliance\best_model_weight.pt')
    # print(model)

    # print(init_log_loss(r'D:\Machine Learning Project\5kCompliance\5kCompliance\log_loss.csv'))
