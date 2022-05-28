import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import time as t
import os
import pickle

from models import NeuralNetwork2
from plot import *
from utils import Time
from module import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

######################################################################

batch_size = 100

# 데이터로더를 생성합니다.
(training_data, test_data) = load_mnist_torch_dataset(flatten=False)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

######################################################################
# 모델 만들기

# 학습에 사용할 CPU나 GPU 장치를 얻습니다.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device\n".format(device))


# 모델을 정의합니다.
model = NeuralNetwork2(save_activation_value=True).to(device)
# model_path = 'model acc_99.49 loss_0.00064153 CCPCPC+CLL(+N,D) 70ep'
# model.load_state_dict(torch.load(model_path+'.pth'))
# model = torch.load('2model acc_99.44 loss_0.00045928 CCPCCPC+CLL(+N,D)'+'.pth')
network = model.__class__.__name__
# print(model)

#####################################################################
# 모델 매개변수 최적화하기

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# show_filters(model)
# show_activation_value_distribution(model, test_dataloader, device, loss_fn, ylim=1e6*2)
# exit()
##############################################################################

graph_datas = {'train_losses':[], 'test_losses':[], 'train_losses_all':[], 'test_accs':[]}
# with open(model_path.replace('model','graph_datas')+'.pkl', 'rb') as f:
    # graph_datas = pickle.load(f)
    # losses = {'train_losses':graph_datas['train_losses'], 'test_losses':graph_datas['test_losses']}
    # plot.loss_graphs(losses, smooth=False, ylim=0.001)
    # plot.accuracy_graph(graph_datas['test_accs'])
    # plot.loss_graph(graph_datas['train_losses_all'], smooth=False, ylim=0.1)
    # exit()

# with open('graph_datas acc_99.42 loss_0.00078264 CCPCPC+CLL(+N,D)'+'.pkl', 'rb') as f:
    # graph_datas2 = pickle.load(f)
    # losses = {'CCPCCPC+CLL(+N,D) loss':graph_datas['test_losses'], 'CCPCPC+CLL(+N,D) loss':graph_datas2['test_losses']}
    # accs = {'CCPCCPC+CLL(+N,D) test_accs':graph_datas['test_accs'], 'CCPCPC+CLL(+N,D) test_accs':graph_datas2['test_accs']}
    # plot.loss_graphs(losses, smooth=False, ylim=0.001)
    # plot.accuracy_graphs(accs)
    # exit()

start_time = t.time()
max_acc = 0
epochs = 10
file_path1, file_path2 = None, None
save_min_acc = 0.994


for i in range(epochs):
    print(f"Epoch {i+1} ({Time.hms_delta(start_time)})\n-------------------------------")
    _, train_loss_avg = train(train_dataloader, model, device, loss_fn, optimizer, graph_datas)
    acc, test_loss_avg = test(test_dataloader, model, device, loss_fn)
    
    graph_datas['train_losses'].append(train_loss_avg)
    graph_datas['test_losses'].append(test_loss_avg)
    graph_datas['test_accs'].append(acc)

    if acc > max_acc:
        max_acc = acc

        if acc > save_min_acc:
            info = f'acc_{(100*acc):>0.2f} loss_{test_loss_avg:>0.8f}'
            # print(file_path1, file_path2)
            if file_path1 != None and os.path.isfile(file_path1): os.remove(file_path1)
            file_path1 = f"model {info} {network}.pth"
            torch.save(model.state_dict(), file_path1)
            torch.save(model, '2'+file_path1)

            if file_path2 != None and os.path.isfile(file_path2): os.remove(file_path2)
            file_path2 = f'graph_datas {info} {network}.pkl'
            with open(file_path2, 'wb') as f:
                pickle.dump(graph_datas, f)

print("Done!")


if epochs > 0:
    with open(f'graph_datas acc_{(100*acc):>0.2f} loss_{test_loss_avg:>0.8f} {network}.pkl', 'wb') as f:
        pickle.dump(graph_datas, f)

    # 모델 저장하기
    save_path = f"model acc_{(100*acc):>0.2f} loss_{test_loss_avg:>0.8f} {network}.pth"
    torch.save(model.state_dict(), save_path)
    print("Saved PyTorch Model State to model.pth")

    losses = {
        'train_losses': graph_datas['train_losses'], 
        'test_losses':graph_datas['test_losses']
    }
    plot.loss_graphs(losses, smooth=False, ylim=0.001)
    plot.accuracy_graph(graph_datas['test_accs'])
    plot.loss_graph(graph_datas['train_losses_all'], ylim=0.1)

#############################################################
# 이제 이 모델을 사용해서 예측을 할 수 있습니다.
plot.set_font(r'font/주아체.ttf')

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
model.eval()

img = np.zeros([28, 28])
# img[8:20,13], img[8:20,14] = 255, 255
img[7:21,7], img[7:21,20], img[7,7:21], img[20,7:21] = 255, 255, 255, 255 # 0
img[14,7:21] = 255
x, y = torch.from_numpy(img.reshape(1, 1, 28, 28)/255).float(), 1

# x, y = test_data[0][0].reshape(1, 1, 28, 28), test_data[0][1]
# with torch.no_grad():
    # pred = model(x)
    # predicted, actual = classes[pred[0].argmax(0)], classes[y]
    # print(f'Predicted: "{predicted}", Actual: "{actual}"')
    # show_wrong_answers_info(x, pred.numpy(), title_info=[f'{actual}, (예측: {predicted})'])

# 틀린 문제들 시각화
wrong_idxs, wrong_ys, wrong_ts = test(test_dataloader, model, device, loss_fn, return_wrong_xs=True)
title_info = [f'{t} | 예측: {y}' for y, t in zip(wrong_ys.argmax(1), wrong_ts)]
print('틀린 문제 수: {}'.format(len(wrong_idxs)))

plot.imgs_show(test_data[wrong_idxs][0], dark_mode=True, text_info=title_info,### training_data -> test_data
    adjust={ 'l': 0,'r': 1,'b': 0.02,'t': 0.98,'hs': 0.05,'ws': 0.02 })

show_wrong_answers_info(test_data[wrong_idxs][0], wrong_ys, title_info='idx', text_info=title_info)

######################################################################
