from torch import nn

ReLU = nn.ReLU()
Dropout = nn.Dropout()


def Conv2d_Norm_ReLU(in_chans, out_chans, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding),
        nn.BatchNorm2d(out_chans),
        ReLU
    )

def Liner_Norm_ReLU(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features),
        ReLU
    )