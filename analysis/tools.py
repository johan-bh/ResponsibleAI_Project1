import torch
import numpy as np

def from_loader_to_tensor(loader,device):
    from tqdm import tqdm

    X = []
    y = []
    a = []
    for batch in tqdm(loader):
        img, lab, a_lab = batch['image'].to(device), batch['label'].to(device), batch['sensitive_attribute'].to(device)
        X.append(img.cpu().numpy())
        y.append(lab.cpu().numpy())
        a.append(a_lab.cpu().numpy())
        # print('debugging..')
        # break

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    a = np.concatenate(a, axis=0)

    # X_test, y_test_, a_test to tensor
    X = torch.tensor(X)
    y = torch.tensor(y)
    a = torch.tensor(a)

    print(f'{X.shape=}, {y.shape=}, {a.shape=}')
    return X, y, a
          