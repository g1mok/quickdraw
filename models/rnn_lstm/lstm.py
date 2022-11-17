'''reference : https://coding-yoon.tistory.com/190'''
import torch
import torch.nn as nn
import ast
import numpy as np

class LSTMnet(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTMnet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3)
        # self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(128*2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
    
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.transpose(x, 1, 2)
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        # print(x.shape)
        # x = self.softmax(x)
        return x

if __name__ == '__main__':

    def pad_sequences(strokes, seq_lengths):
        seq_tensor = np.zeros((seq_lengths, 3))
        for idx, stroke in enumerate(strokes):
            seq_tensor[idx, :] = stroke
        return seq_tensor

    arrs = [[[184, 115, 67, 57, 36, 18], [251, 103, 12, 109, 193, 247]], [[145, 154, 150, 48, 24], [180, 176, 175, 178, 173]], [[55, 50, 38, 0, 28, 31, 60, 66, 71], [128, 152, 179, 253, 254, 247, 114, 26, 0]], [[72, 76, 81, 83, 86, 99, 93, 86], [26, 35, 71, 250, 255, 254, 189, 5]], [[67, 74, 97, 172], [12, 16, 49, 194]]]

    arrs = str(arrs)
    stroke_vec = ast.literal_eval(arrs)
    in_strokes = [(xi,yi,i) for i,(x,y) in enumerate(stroke_vec) for xi,yi in zip(x,y)]
    c_strokes = np.stack(in_strokes)
    c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
    c_strokes[:,2] += 1
    # c_strokes= c_strokes.swapaxes(0, 1)
    # print("swap", c_strokes)
    padding_stroke = torch.Tensor(pad_sequences(c_strokes, 70))
    padding_stroke = padding_stroke.unsqueeze(0)
    print(padding_stroke.shape)
    n_class = 345
    net = LSTMnet(input_size=3, output_size=n_class)
    output = net(padding_stroke)
    print(output.shape)
    