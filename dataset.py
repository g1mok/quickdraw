import os
import cv2
import pandas as pd
import ast
import numpy as np
import torch.nn
from torch.utils.data import Dataset, DataLoader
from utils import *


class Customdataset(Dataset):
    def __init__(self, dir_path, files_list, label_list, n_class, mode):
        super().__init__()
        self.dir_path = dir_path
        self.files_list = files_list
        self.label_list = label_list
        self.length_files = len(files_list)

        self.one_hot = torch.nn.functional.one_hot(torch.arange(n_class))

        self.mode = mode

    def __len__(self):
        return self.length_files

    def __getitem__(self, idx):
        file = self.files_list[idx]
        file_path = os.path.join(self.dir_path, file)

        label = self.label_list[idx]
        idx = classes.index(label)
        label = self.one_hot[idx].float()

        data = pd.read_csv(file_path)
        new_coords = ast.literal_eval(data['drawing'].values[0])

        size = 224
        if self.mode == 'weight':
            x = draw_cv2_weight(new_coords, size=size)
        else:
            x = draw_cv2img(new_coords, size=size)

        image = torch.from_numpy(x / 255).permute(2, 0, 1).float()

        stroke_vec = [(xi,yi,i) for i,(x,y) in enumerate(new_coords) for xi,yi in zip(x,y)]
        c_strokes = np.stack(stroke_vec)
        c_strokes[:,2] = [1]+np.diff(c_strokes[:,2]).tolist()
        c_strokes[:,2] += 1

        padding_stroke = torch.Tensor(pad_sequences(c_strokes, 70))

        return image, padding_stroke, label

        
if __name__=='__main__':
    dir_path = './tinyquickdraw_data/key_data'
    files_list = os.listdir(dir_path)[4980:5025]

    dataset = Customdataset(dir_path, files_list, n_class)
    dataloader = DataLoader(dataset, batch_size=4)
    for i, data in enumerate(dataloader):
        image, stroke, label = data
        print(image.size(0), stroke.shape)
        break