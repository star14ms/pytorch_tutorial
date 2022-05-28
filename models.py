import torch
from torch import nn
from layers import (
    Conv2d_Norm_ReLU,
    Liner_Norm_ReLU,
    ReLU, 
    Dropout,
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            Conv2d_Norm_ReLU(1, 16),
            Conv2d_Norm_ReLU(16, 32),
            nn.Flatten(),

            nn.Linear(32*28*28, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(50, 10),
            nn.Dropout(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# CCPCc+CLL(+N,D)
class NeuralNetwork1(nn.Module):
    def __init__(self, img_size=28, c1=32, c2=32, c3=64, c4=64, hiddens=50, save_activation_value=False):
        super().__init__()
        img_size = (28//2+2)//2
        in_features = (c3+c4)*img_size*img_size

        self.conv1_pool = nn.Sequential(
            Conv2d_Norm_ReLU(1, c1), 
            Conv2d_Norm_ReLU(c1, c1), 
            nn.MaxPool2d(kernel_size=2, stride=2), # /2
        )
        self.conv2 = Conv2d_Norm_ReLU(c1, c2, kernel_size=2, stride=2) # /2+1
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

        self.save_activation_value = save_activation_value
        
        if save_activation_value:
            self.init_saved_act_values()
            
    def forward(self, x):
        x1 = self.conv1_pool(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat([x3, x4], dim=1)
        x = self.flatten(x)

        x5 = self.liner1(x)
        y = self.liner2(x5)
        
        if self.save_activation_value:
            self.act_values['conv1'].append(torch.clone(x1).cpu())
            self.act_values['conv2'].append(torch.clone(x2).cpu())
            self.act_values['conv3'].append(torch.clone(x3).cpu())
            self.act_values['conv4'].append(torch.clone(x4).cpu())
            self.act_values['liner1'].append(torch.clone(x5).cpu())

        return y

    def init_saved_act_values(self):
        self.act_values = {'conv1':[], 'conv2':[], 'conv3':[], 'conv4':[], 'liner1':[]}
    
    def set_save_activation_value(self, save_activation_value):
        self.save_activation_value = save_activation_value
        if not save_activation_value:
            self.init_saved_act_values()


# CCPCPC+CLL(+N,D)
class NeuralNetwork2(nn.Module):
    def __init__(self, img_size=28, c1=32, c2=64, c3=64, c4=64, hiddens=50, save_activation_value=False):
        super().__init__()
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

        self.save_activation_value = save_activation_value
        
        if save_activation_value:
            self.init_saved_act_values()
            
    def forward(self, x):
        x1 = self.conv1_pool(x)
        x2 = self.conv2_pool(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat([x3, x4], dim=1)
        x = self.flatten(x)

        x5 = self.liner1(x)
        y = self.liner2(x5)

        if self.save_activation_value:
            self.act_values['conv1'].append(torch.clone(x1).cpu())
            self.act_values['conv2'].append(torch.clone(x2).cpu())
            self.act_values['conv3'].append(torch.clone(x3).cpu())
            self.act_values['conv4'].append(torch.clone(x4).cpu())
            self.act_values['liner1'].append(torch.clone(x5).cpu())

        return y

    def init_saved_act_values(self):
        self.act_values = {'conv1':[], 'conv2':[], 'conv3':[], 'conv4':[], 'liner1':[]}
    
    def set_save_activation_value(self, save_activation_value):
        self.save_activation_value = save_activation_value
        if not save_activation_value:
            self.init_saved_act_values()


# CCPCCPC+CLL(+N,D)
class NeuralNetwork3(nn.Module):
    def __init__(self, img_size=28, c1=32, c2=64, c3=64, c4=64, hiddens=50, save_activation_value=False):
        super().__init__()
        img_size = (28//2+2)//2
        in_features = (c3+c4)*img_size*img_size

        self.conv1_pool = nn.Sequential(
            Conv2d_Norm_ReLU(1, c1), 
            Conv2d_Norm_ReLU(c1, c1), 
            nn.MaxPool2d(kernel_size=2, stride=2), # /2
        )
        self.conv2_pool = nn.Sequential(
            Conv2d_Norm_ReLU(c1, c2), # +2
            Conv2d_Norm_ReLU(c2, c2, padding=2),
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
        
        self.save_activation_value = save_activation_value
        
        if save_activation_value:
            self.init_saved_act_values()
            
    def forward(self, x):
        x1 = self.conv1_pool(x)
        x2 = self.conv2_pool(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat([x3, x4], dim=1)
        x = self.flatten(x)

        x5 = self.liner1(x)
        y = self.liner2(x5)

        if self.save_activation_value:
            self.act_values['conv1'].append(torch.clone(x1).cpu())
            self.act_values['conv2'].append(torch.clone(x2).cpu())
            self.act_values['conv3'].append(torch.clone(x3).cpu())
            self.act_values['conv4'].append(torch.clone(x4).cpu())
            self.act_values['liner1'].append(torch.clone(x5).cpu())

        return y

    def init_saved_act_values(self):
        self.act_values = {'conv1':[], 'conv2':[], 'conv3':[], 'conv4':[], 'liner1':[]}
    
    def set_save_activation_value(self, save_activation_value):
        self.save_activation_value = save_activation_value
        if not save_activation_value:
            self.init_saved_act_values()
