import torch
from torch import nn
import torch.nn.functional as F

class ResNetBaselineNew(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        # self.layers = nn.Sequential(*[
        #     ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
        #     ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
        #     ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        # ])
        self.block1 = ResNetBlock(in_channels=in_channels, out_channels=mid_channels)
        self.block2 = ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2)
        self.block3 = ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2)
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False, context=None) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        # return self.final(x.mean(dim=-1))
        x= torch.swapaxes(x, 1,2) # B, C, T
        x1=self.block1(x)
        x2=self.block2(x1)
        x3=self.block3(x2)

        if context==None:
            y=x3.mean(dim=-1)
        else:
            y=context

        if get_ha:
            return self.final(y), y, y, y,y
        return self.final(y)
        
class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x, get_ha=False, context=None):
        #if last hidden state is provided pass though the classifier directly , if not pass through rnn to get context vector first
        if context==None:
            h0, c0 = self.init_hidden(x)
            out_t, (hn, cn) = self.rnn(x, (h0, c0))
            context = out_t[:, -1, :]
            out = self.fc(context)
        else:
            out, out_t, hn, cn = self.fc(context), None, None, None
        
        if get_ha:
            return out, context, out_t, hn, cn
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

class LSTMClassifierFullContext(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x, get_ha=False, context=None):
        #if last hidden state is provided pass though the classifier directly , if not pass through rnn to get context vector first
        if context==None:
            h0, c0 = self.init_hidden(x)
            out_t, (hn, cn) = self.rnn(x, (h0, c0))
            context = out_t
            out = self.fc(context[:, -1, :])
        else:
            out, out_t, hn, cn = self.fc(context[:, -1, :]), None, None, None
        
        if get_ha:
            return out, context, out_t, hn, cn
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]
        
class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)
class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)


class ResNetBaseline(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        # self.layers = nn.Sequential(*[
        #     ResNetBlock(in_channels=in_channels, out_channels=mid_channels),
        #     ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
        #     ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        # ])
        self.block1 = ResNetBlock(in_channels=in_channels, out_channels=mid_channels)
        self.block2 = ResNetBlock(in_channels=mid_channels, out_channels=mid_channels * 2)
        self.block3 = ResNetBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2)
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        # return self.final(x.mean(dim=-1))
        x1=self.block1(x)
        x2=self.block2(x1)
        x3=self.block3(x2)
        if get_ha:
            return self.final(x3.mean(dim=-1)),x1, x2,x3
        return self.final(x3.mean(dim=-1))
        


class ResNetBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class ResNetBaseline2(nn.Module):
    """A PyTorch implementation of the ResNet Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    mid_channels:
        The 3 residual blocks will have as output channels:
        [mid_channels, mid_channels * 2, mid_channels * 2]
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        # self.layers = nn.Sequential(*[
        #     ResNetBlock2(in_channels=in_channels, out_channels=mid_channels),
        #     ResNetBlock2(in_channels=mid_channels, out_channels=mid_channels * 2)

        # ])
        self.block1 = ResNetBlock2(in_channels=in_channels, out_channels=mid_channels)
        self.block2 = ResNetBlock2(in_channels=mid_channels, out_channels=mid_channels * 2)
        self.final = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        # return self.final(x.mean(dim=-1))
        x1=self.block1(x)
        x2=self.block2(x1)
        if get_ha:
            return self.final(x2.mean(dim=-1)),x1, x2
        return self.final(x2.mean(dim=-1))

class ResNetBlock2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels]
        kernel_sizes = [5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)

class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        # self.layers = nn.Sequential(*[
        #     ConvBlock(in_channels, 128, 8, 1),
        #     ConvBlock(128, 256, 5, 1),
        #     ConvBlock(256, 128, 3, 1),
        # ])
        #added for intermediate feature extarction
        self.layer1 = ConvBlock(in_channels, 128, 8, 1)
        self.layer2 = ConvBlock(128, 256, 5, 1)
        self.layer3 =  ConvBlock(256, 128, 3, 1)

        # self.linear1 = nn.linear(128,128)# to get linear embeddings
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        x1= self.layer1(x)
        x2= self.layer2(x1)
        x3= self.layer3(x2)
        # x4= self.linear1(x3)
        linear = self.final(x3.mean(dim=-1))

        if get_ha:
            return x1, x2, x3, x3.mean(dim=-1),linear
            
        return linear
        
class FCNBaselineSmall(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int =8, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        # self.layers = nn.Sequential(*[
        #     ConvBlock(in_channels, 128, 8, 1),
        #     ConvBlock(128, 256, 5, 1),
        #     ConvBlock(256, 128, 3, 1),
        # ])
        #added for intermediate feature extarction
        self.layer1 = ConvBlock(in_channels, mid_channels, 8, 1)
        self.layer2 = ConvBlock(mid_channels, mid_channels*2, 5, 1)
        self.layer3 =  ConvBlock(mid_channels*2, mid_channels, 3, 1)
        self.final = nn.Linear(mid_channels, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        x1= self.layer1(x)
        x2= self.layer2(x1)
        x3= self.layer3(x2)
        linear = self.final(x3.mean(dim=-1))

        if get_ha:
            return linear, x1, x2, x3
            
        return linear

class FCNBaselineSmall2(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, mid_channels: int =8, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        # self.layers = nn.Sequential(*[
        #     ConvBlock(in_channels, 128, 8, 1),
        #     ConvBlock(128, 256, 5, 1),
        #     ConvBlock(256, 128, 3, 1),
        # ])
        #added for intermediate feature extarction
        self.layer1 = ConvBlock(in_channels, mid_channels, 5, 1)
        self.layer2 = ConvBlock(mid_channels, mid_channels*2, 3, 1)
        # self.layer3 =  ConvBlock(mid_channels*2, mid_channels, 3, 1)
        self.final = nn.Linear(mid_channels*2, num_pred_classes)

    def forward(self, x: torch.Tensor, get_ha=False) -> torch.Tensor:  # type: ignore
        # x = self.layers(x)
        x1= self.layer1(x)
        x2= self.layer2(x1)
        # x3= self.layer3(x2)
        linear = self.final(x2.mean(dim=-1))

        if get_ha:
            return linear, x1, x2, 0
            
        return linear

class LSTMClassifier_old(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x, get_ha=False):
        h0, c0 = self.init_hidden(x)
        out_t, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out_t[:, -1, :])
        if get_ha:
            return out, out_t, hn, cn
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]

class LSTMClassifierTeacherUnrolled(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn1 = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.rnn3 = nn.LSTM(hidden_dim, hidden_dim, 1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
    
    def forward(self, x, get_ha=False):
        h0, c0 = self.init_hidden(x)
        out_t1, (hn1, cn1) = self.rnn1(x, (h0, c0))
        out_t2, (hn2, cn2) = self.rnn2(out_t1, (hn1, cn1))
        out_t3, (hn3, cn3) = self.rnn3(out_t2, (hn2, cn2))
        out = self.fc(out_t3[:, -1, :])
        if get_ha:
            return out, out_t1, out_t2, out_t3
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim)
        return [t.cuda() for t in (h0, c0)]



class FCNBaseline_orig(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        ])
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.layers(x)
        return self.final(x.mean(dim=-1))