from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, x, t, transform=None, target_transform=None):
        self.x = x
        self.t = t
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.t[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label,)
        return sample



import torch
import numpy as np
from dataset.mnist import load_mnist


def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t


def load_mnist_torch_dataset(normalize=True, flatten=True, one_hot_label=False, train_sample='all', test_sample='all', shuffle=True):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=normalize, flatten=flatten, one_hot_label=one_hot_label)
    
    if shuffle:
        x_train, t_train = shuffle_dataset(x_train, t_train)
        x_test, t_test = shuffle_dataset(x_test, t_test)

    x_train, x_test = torch.from_numpy(x_train), torch.from_numpy(x_test)
    t_train, t_test = torch.from_numpy(t_train.astype(np.int64)), torch.from_numpy(t_test.astype(np.int64))
    
    if train_sample != 'all': 
        x_train, t_train = x_train[:train_sample], t_train[:train_sample]
    if test_sample != 'all': 
        x_test, t_test = x_test[:test_sample], t_test[:test_sample]

    training_data = CustomImageDataset(x_train, t_train)
    test_data = CustomImageDataset(x_test, t_test)

    return (training_data, test_data)


def train(dataloader, model, device, loss_fn, optimizer, graph_datas=None, verbose=True):
    size = len(dataloader.dataset)
    train_loss, correct = 0, 0
    for batch, (X, t) in enumerate(dataloader):
        X, t = X.to(device), t.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, t)
        if graph_datas!=None: graph_datas['train_losses_all'].append(loss)
        train_loss += loss_fn(pred, t).item()
        correct += (pred.argmax(1) == t).type(torch.float).sum().item()

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            if verbose: print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= size
    correct /= size
    return correct, train_loss


def test(dataloader, model, device, loss_fn, return_wrong_xs=False, verbose=True):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    if return_wrong_xs: wrong_idxs, wrong_ys, wrong_ts = [], [], []
    with torch.no_grad():
        for X, t in dataloader:
            X, t = X.to(device), t.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, t).item()
            correct += (pred.argmax(1) == t).type(torch.float).sum().item()
            if return_wrong_xs: 
                wrong_idxs.extend(pred.argmax(1) != t)
                wrong_ys.extend(pred.numpy())
                wrong_ts.extend(t)

    test_loss /= size
    correct /= size
    if verbose: print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")

    if return_wrong_xs:
        wrong_idxs = np.where(np.array(wrong_idxs) == True)[0]
        wrong_ys = np.array(wrong_ys)[wrong_idxs]
        wrong_ts = np.array(wrong_ts)[wrong_idxs]
        return wrong_idxs, wrong_ys, wrong_ts
    else:
        return correct, test_loss



from plot import plot
import sys


def show_filters(model):
    for param in model.parameters():
        print(param.size(), 'Conv params!!' if len(param.size()) == 4 else '') 
        if len(param.size()) == 4: plot.imgs_show(param.detach().numpy())


def show_activation_value_distribution(model, test_dataloader, device, loss_fn, ylim=1e6*2):
    test(test_dataloader, model, device, loss_fn, return_wrong_xs=False)
    
    act_values = model.act_values

    for key in act_values.keys():
        while len(act_values[key]) != 1:
            act_values[key][0] = torch.cat([act_values[key][0], act_values[key][1]], dim=1)

            sys.stdout.write(f'\r{act_values[key][0].size()} {act_values[key][1].size()}')
            sys.stdout.flush()

            del act_values[key][1]
            
        act_values[key] = act_values[key][0].numpy()
        print()
    
    plot.activation_value_distribution(act_values, ylim=ylim)
    plot.activation_value_distribution(act_values)


import matplotlib.pyplot as plt
import torch.nn.functional as F


def show_wrong_answers_info(wrong_xs, wrong_ys, dark_mode=True, title='', title_info=[], text_info=[]):
    len_wrongs = len(wrong_ys)
    cmap='gray' if dark_mode else plt.cm.gray_r
    size = 4 if len_wrongs != 1 else 2
    bar1_idxs = [2, 4, 10, 12] if len_wrongs != 1 else [2]
    bar2_idxs = [6, 8, 14, 16] if len_wrongs != 1 else [4]

    plots = 0
    for i, wrong_x, wrong_y in zip(range(len_wrongs), wrong_xs, wrong_ys):
        if plots == 0: 
            fig = plt.figure()
            fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.92, hspace=0.25, wspace=0.25)
        plots += 1

        plt.rcParams["font.size"] = 20
        ax = fig.add_subplot(size//2, size, 2*plots-1, xticks=[], yticks=[])
        if title_info!=[]:
            info = title_info[i] if title_info!='idx' else i+1
            if title=='':
                plt.title(f'{i+1}/{len_wrongs}\n{info}')
            else:
                plt.title(f'{i+1}/{len_wrongs}\n{title} ({info})') ### plot() 후에 나와야 함

        if text_info!=[]:
            ax.text(0, 0, text_info[i], ha="left", va="top", color='white')
            fig.canvas.draw()

        ax.imshow(wrong_x[0], cmap=cmap, interpolation='nearest')

        plt.rcParams["font.size"] = 11
        x = np.arange(10)
        y = wrong_y
        ax = fig.add_subplot(size, size, bar1_idxs[plots-1], xticks=x, yticks=np.round(sorted(y), 1), xlabel='손글씨 숫자 예측 | 위: 점수 | 아래: 확률(%)')
        ax.bar(x, y)
        
        if type(wrong_y) == np.ndarray: wrong_y = torch.from_numpy(wrong_y)
        y = F.softmax(wrong_y, dim=0)*100
        ax = fig.add_subplot(size, size, bar2_idxs[plots-1], xticks=x, yticks=sorted(y)[8:], ylim=(0, 100))
        ax.bar(x, y)
        
        if (i+1)%4 == 0 or i == len_wrongs-1: 
            plots = 0
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.pause(0.01)
            plt.show()
