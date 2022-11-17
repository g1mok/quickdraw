import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from models.mobilenetv2 import mobnet
from models.efficientnet import effinet
from efficientnet_pytorch import EfficientNet
from torchvision.models import efficientnet_b0
from dataset import Customdataset

from tqdm import tqdm
from utils import *

import ast
import pandas as pd


def inference(model, model_path, image):
    saved_pth_dir = './save_models'
    state = torch.load(os.path.join(saved_pth_dir, model_path))
    model.load_state_dict(state['model'])
    model.eval()

    image = cv2.resize(image, (224, 224))
    image = torch.from_numpy(image / 255).permute(2, 0, 1).unsqueeze(0).float().to(device)

    output = model(image)
    _, y_pred = torch.max(output, 1)
    predict_cls = y_pred[0].cpu()
    print(classes[predict_cls])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mobnet.mobilenet_v2(n_class=n_class, pretrained=False).to(device)
    model_path = 'mobilenet_33&0.094.pth'
    data_path = './quickdraw_data/key_data/angel_4864577625915392.csv'

    data = pd.read_csv(data_path)
    new_coords = ast.literal_eval(data['drawing'].values[0])
    image = draw_cv2img(new_coords, size=224)

    inference(model, model_path, image)