import torch
import torch.nn.functional as F

num_layers = 2
hidden_size = 16
input_size = 1

conv1_size = 64
conv2_size = 128
conv3_size = 256
conv4_size = 384
conv5_size = 512

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention_weights = None
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),  # hidden_dim 为lstm的隐藏层数
            torch.nn.Tanh(True),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: (batch_size, sequence_length, hidden_dim)
        # 计算注意力得分，赋予权重
        energy = self.projection(encoder_outputs)  # (batch_size, sequence_length, 1)
        weights = F.softmax(energy.squeeze(-1), dim=1)  # (batch_size, sequence_length)

        # 应用注意力权重
        outputs = (encoder_outputs * weights.unsqueeze(-1))  # (batch_size, sequence_length, hidden_dim)
        self.attention_weights = weights
        return outputs


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_1 = torch.nn.Conv1d(in_channels=hidden_size, out_channels=conv1_size, kernel_size=3, padding=1,
                                      stride=2, bias=False)
        self.conv_2 = torch.nn.Conv1d(in_channels=conv1_size, out_channels=conv2_size, kernel_size=3, padding=1,
                                      stride=2, bias=False)
        self.conv_3 = torch.nn.Conv1d(in_channels=conv2_size, out_channels=conv3_size, kernel_size=3, padding=1,
                                      stride=2, bias=False)
        self.conv_4 = torch.nn.Conv1d(in_channels=conv3_size, out_channels=conv4_size, kernel_size=3, padding=1,
                                      stride=2, bias=False)
        self.conv_5 = torch.nn.Conv1d(in_channels=conv4_size, out_channels=conv5_size, kernel_size=3, padding=1,
                                      stride=2, bias=False)

        self.avgpool = torch.nn.AvgPool1d(kernel_size=2)

        self.bn_1 = torch.nn.BatchNorm1d(conv1_size)
        self.bn_2 = torch.nn.BatchNorm1d(conv2_size)
        self.bn_3 = torch.nn.BatchNorm1d(conv3_size)
        self.bn_4 = torch.nn.BatchNorm1d(conv4_size)
        self.bn_5 = torch.nn.BatchNorm1d(conv5_size)

        self.adp_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.01)
        self.attention = SelfAttention(hidden_size)

        self.fc_1 = torch.nn.Linear(in_features=conv5_size, out_features=conv2_size)
        self.fc_2 = torch.nn.Linear(in_features=conv2_size, out_features=1)

        self.relu = torch.nn.ReLU(inplace=True)

        self.dropout = torch.nn.Dropout1d(p=0.02)

    def forward(self, x):  # (bs, 1000, 1)
        hidden_0 = torch.zeros(num_layers, x.shape[0], hidden_size, device=device)  # (nl, bs, hidden_size)
        cell_0 = torch.zeros_like(hidden_0, device=device)                          # (nl, bs, hidden_size)

        # 先LSTM后CNN
        lstm_out, (hidden_n, _) = self.lstm(x, (hidden_0, cell_0))  # (bs, 2000, 16)

        # 自注意力
        self_att_out = self.attention(lstm_out)                 # (bs, 2000, 16)
        lstm_attention = self_att_out.transpose(-1, -2)         # (bs, 16, 2000)

        x1 = self.relu(self.bn_1(self.conv_1(lstm_attention)))  # (bs, 64, 1000)
        x1 = self.avgpool(x1)                                   # (bs, 64, 500)
        x2 = self.relu(self.bn_2(self.conv_2(x1)))              # (bs, 128, 250)
        x2 = self.avgpool(x2)                                   # (bs, 128, 125)
        x3 = self.relu(self.bn_3(self.conv_3(x2)))              # (bs, 256, 62)
        x4 = self.relu(self.bn_4(self.conv_4(x3)))              # (bs, 384, 31)
        x5 = self.relu(self.bn_5(self.conv_5(x4)))              # (bs, 512, 15)

        x6 = self.adp_avgpool(x5)                               # (bs, 384, 1)
        x7 = torch.flatten(x6, start_dim=1)                     # (bs, 384)
        x8 = self.dropout(self.relu(self.fc_1(x7)))             # (bs, 128)
        x9 = self.fc_2(x8)                                      # (bs, 1)

        return x9
