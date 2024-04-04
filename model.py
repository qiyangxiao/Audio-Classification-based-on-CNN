import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class AudioMutiCNN(nn.Module):
    def __init__(self, config):
        super(AudioMutiCNN, self).__init__()
        self.num_classes = config['num_classes']
        self.time_length = config['time_length']
        self.input_dim = config['input_dim']
        self.conv1_outdim = config['conv1_outdim']
        self.conv2_outdim = config['conv2_outdim']
        self.conv3_outdim = config['conv3_outdim']
        self.conv4_outdim = config['conv4_outdim']
        self.dropout = config['dropout']
        self.kernel_size = config['kernel_size']
        self.stride = config['stride']
        self.padding = config['padding']

        conv_layers = []
        # First CNN
        self.conv1 = nn.Conv1d(self.input_dim, self.conv1_outdim, kernel_size=self.kernel_size,
                               stride=self.stride)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.conv1_outdim)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()

        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second CNN
        self.conv2 = nn.Conv1d(self.conv1_outdim, self.conv2_outdim, kernel_size=self.kernel_size,
                               stride=self.stride)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(self.conv2_outdim)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv2.bias.data.zero_()

        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third CNN
        self.conv3 = nn.Conv1d(self.conv2_outdim, self.conv3_outdim, kernel_size=self.kernel_size,
                               stride=self.stride)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(self.conv3_outdim)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()

        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth CNN
        self.conv4 = nn.Conv1d(self.conv3_outdim, self.conv4_outdim, kernel_size=self.kernel_size,
                               stride=self.stride)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(self.conv4_outdim)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()

        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Linear Classifier
        self.fc = nn.Linear(in_features=self.conv4_outdim, out_features=self.num_classes)

        self.conv = nn.Sequential(*conv_layers)

        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        x = self.conv(x)

        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)

        out = self.fc(self.dropout_layer(x))
        
        return F.log_softmax(out, dim=1)