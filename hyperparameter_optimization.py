import time as t
import numpy as np
import sys
sys.path.extend(['.', r'C:\Users\danal\Documents\programing\python'])
from util import time, Alerm
from plot import plot
import pickle, os
from torch.utils.data import DataLoader
from module import load_mnist_torch_dataset, Conv2d_Norm_ReLU, Liner_Norm_ReLU, Dropout, train, test
from torch import nn
import torch

pkl_file = None
# pkl_file = "hyperparameter_optimization_info" + ".pkl"
if pkl_file != None:
    with open(pkl_file, 'rb') as f:
        r = pickle.load(f)
    plot.many_accuracy_graphs(r["results_train"], r["results_val"], graph_draw_num=20, col_num=5)
    plot.many_loss_graphs(r["results_losses"], graph_draw_num=20, col_num=5, verbose=False, smooth=False)
    plot.many_loss_graphs(r["results_losses"], graph_draw_num=20, col_num=5, verbose=False, smooth=True)
    exit()


# 학습할 데이터 만들기
(training_data, test_data) = load_mnist_torch_dataset(flatten=False, train_sample=1000, test_sample=1000)
train_dataloader = DataLoader(training_data, batch_size=100)
test_dataloader = DataLoader(test_data, batch_size=100)


# 네트워크 만들기
class NeuralNetwork2(nn.Module): # CCPCPC+CLL(+N,D)
    def __init__(self, img_size=28, c1=32, c2=64, c3=64, c4=64, hiddens=50):
        super(NeuralNetwork2, self).__init__()
        img_size = (28//2+2)//2
        in_features = (c3+c4)*img_size*img_size

        self.conv1_pool = nn.Sequential(
            Conv2d_Norm_ReLU(1, c1), 
            Conv2d_Norm_ReLU(c1, c1), 
            nn.MaxPool2d(kernel_size=2, stride=2), # /2
        )
        self.conv2_pool = nn.Sequential(
            Conv2d_Norm_ReLU(c1, c2, padding=2), # +2
            nn.MaxPool2d(kernel_size=2, stride=2), # /2
        )
        self.conv3 = Conv2d_Norm_ReLU(c2, c3)
        self.conv4 = Conv2d_Norm_ReLU(c3, c4)

        self.flatten = nn.Sequential(
            nn.Flatten(),
            Dropout,
        )   
        self.liner1 = nn.Sequential(
            Liner_Norm_ReLU(in_features, hiddens),
            Dropout,
        )
        self.liner2 = nn.Sequential(
            nn.Linear(hiddens, 10),
            Dropout,
        )

    def forward(self, x):
        x1 = self.conv1_pool(x)
        x2 = self.conv2_pool(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat([x3, x4], dim=1)
        x = self.flatten(x)

        x5 = self.liner1(x)
        y = self.liner2(x5)

        # if save_activation_value:
            # act_values['conv1'].append(x1)
            # act_values['conv2'].append(x2)
            # act_values['conv3'].append(x3)
            # act_values['conv4'].append(x4)
            # act_values['liner1'].append(x5)

        return y

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = 'model hyperparameter_optimization'
loss_fn = nn.CrossEntropyLoss()

# 시험할 최적화 방법 종류와 학습률, 가중치 감소 계수 범위 지정 (log 스케일 10^n)
optimizers = {
    # 'SGD':      {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
    # 'Momentum': {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
    # 'Adagrad':  {"lr_min": -1, "lr_max": -5, "wd_min": -8, "wd_max": -4}
    'Adam':     {"lr_min": -1, "lr_max": -4, "wd_min": -8, "wd_max": -4}
    # 'Nesterov': {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
    # 'Rmsprop':  {"lr_min": -3, "lr_max": -1, "wd_min": -8, "wd_max": -4}
}
optimization_trial = 50
attempts_number = 1
epochs = 10

results_losses = {}
results_train = {}
results_val = {}
# give_up = {'epoch': 3} # 'test_acc':0.1
digit = 6


# 학습 정보 출력 함수 정의
def print_learning_info(str_lr, str_wd, val_accs, attempts_number, isgiveup, good_acc=95):
    
    average_acc = sum(val_accs) / len(val_accs)          
    
    accuracy_info = "acc: None"
    if not isgiveup:
        accuracy_info = ("acc: %.1f%%" % average_acc).rjust(5)
    
    str_margin_of_error95 = ""
    if attempts_number > 1:
        standard_deviation = (sum( (np.array(val_accs)-average_acc)**2 ) / len(val_accs)) ** (1/2) ### ** 우선도가 /보다 높음
        standard_error = standard_deviation / (len(val_accs) ** (1/2)) 
        margin_of_error95 = round(1.96 * standard_error, 2)

        str_margin_of_error95 = ("(±%.2f%%)" % margin_of_error95).rjust(5)
    
    print(f"lr: {str_lr.rjust(digit+2)} | wd⁁: {str_wd} | {accuracy_info} {str_margin_of_error95}", end=" ")
    
    if isgiveup:
        print(f"-- I gave up")
    elif average_acc > good_acc:
        print("-- good!! 💚")
    else:
        print()

print(f"trial:{optimization_trial}, attempts:{attempts_number}, epochs:{epochs}")
print("\n탐색 시작!!")
start_time = t.time()

# 매개변수 최적화 방법과 학습률에 따른 딥러닝 효율 비교
for optimizer in optimizers:

    for i in range(optimization_trial):
        lr = 10 ** np.random.uniform(optimizers[optimizer]["lr_min"], optimizers[optimizer]["lr_max"])
        wd = 10 ** np.random.uniform(optimizers[optimizer]["wd_min"], optimizers[optimizer]["wd_max"])

        val_accs = []
        for attempts in range(attempts_number):
            model = NeuralNetwork2().to(device)
            optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            graph_datas = {'train_losses':[], 'test_losses':[], 'train_losses_all':[], 'train_accs':[], 'test_accs':[]}

            for epoch in range(epochs):
                train_acc_avg, _ = train(train_dataloader, model, device, loss_fn, optim, graph_datas=None, verbose=False)
                test_acc_avg, test_loss_avg = test(test_dataloader, model, device, loss_fn, return_wrong_xs=False, verbose=False)

                graph_datas['test_losses'].append(test_loss_avg)
                graph_datas['train_accs'].append(train_acc_avg*100)
                graph_datas['test_accs'].append(test_acc_avg*100)

                sys.stdout.write(f'\repoch: {epoch} ({time.str_hms_delta(start_time)})')
                sys.stdout.flush()
            print('\r')

            str_lr = str(np.format_float_scientific(wd, precision=2)) # f"%.{digit}f"% lr ### %{digit}.f
            str_wd = str(np.format_float_scientific(wd, precision=2)) # 
            key = (f"{optimizer} | " if len(optimizers) > 1 else "") + f"lr: {str_lr} | wd: {str_wd}" + (f" | {attempts}" if attempts > 1 else "")
            
            results_losses[key] = graph_datas['test_losses']
            results_train[key] = graph_datas['train_accs'] ### *100 하면 100번 복사됨
            results_val[key] = graph_datas['test_accs']
            val_accs.append(graph_datas['test_accs'][-1])
        
        if i == 0 and optimizer == list(optimizers.keys())[0]:
            first_trial_time = int(t.time() - start_time)
            estimated_time = first_trial_time * optimization_trial * attempts_number * len(optimizers)
            print(f"예상 소요 시간: {time.str_hms(estimated_time)}")
        if i == 0: 
            print("\noptimizer: " + optimizer)
        print(f"{i+1}".rjust(2)+"/"+f"{optimization_trial}".rjust(2), end=" | ")
        print_learning_info(str_lr, str_wd, val_accs, attempts_number, isgiveup=False)

# 학습 끝나면 소요 시간 출력하고, 정보 저장 후, 알람 울리기
print(time.str_hms_delta(start_time))
# Alerm()

results = {"results_train":results_train, "results_val":results_val, "results_losses":results_losses}
with open('hyperparameter_optimization_info.pkl', 'wb') as f:
        pickle.dump(results, f)
print(f"{pkl_file} 저장 성공!")

print("=========== Hyper-Parameter Optimization Result ===========")
plot.many_accuracy_graphs(results_train, results_val, graph_draw_num=20, col_num=5, sort=True)
plot.many_loss_graphs(results_losses, graph_draw_num=20, col_num=5, sort=True, verbose=False, smooth=False)
plot.many_loss_graphs(results_losses, graph_draw_num=20, col_num=5, sort=True, verbose=False, smooth=False)
