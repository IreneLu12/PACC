import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import joblib

from model.densenet121 import PVQS_extractor
from utils.dataloader import frame_loader

patch_loc = [[55,355],[55, 355 + 299 + 10],[55 + 10 + 299,355],[55 + 10 + 299,355 + 299 + 10]]


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = PVQS_extractor()
    feature_extractor.load_state_dict(torch.load("./file/DenseNet121.pth"))
    feature_extractor.to(device)
    feature_extractor.eval()
    classifier = joblib.load("./file/classifier")
    frame_dataset = frame_loader("./file/info.txt")
    data_loader = DataLoader(frame_dataset,
                           batch_size=1,
                           shuffle=False,
                           num_workers=4)

    patch = torch.zeros((4,3,299,299)).to(device)
    correct_count = 0
    for i, data in enumerate(data_loader):
        input, label = data
        label = label.data.cpu().numpy()
        for i,item in enumerate(patch_loc):
            patch[i] = input[0,:,item[0]:item[0] + 299,item[1]:item[1] + 299]
        output1,output2 = feature_extractor(patch)
        output1 = output1.mean(dim=0)
        output2 = output2.mean(dim=0)
        output = torch.cat([output1,output2],dim=0)
        output = output.unsqueeze(0)
        if device == torch.device("cuda:0"):
            output = output.data.cpu().numpy()
        else:
            output = output.data.numpy()
        pred_level = classifier.predict(output)
        if pred_level == label:
            correct_count += 1
    print("The predicted accuracy in demo frames set: ",correct_count / len(data_loader))


if __name__ == "__main__":
    test()