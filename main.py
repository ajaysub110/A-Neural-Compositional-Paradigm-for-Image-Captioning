import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn 
from modules import resnet

def main():
    model = resnet.resnet152(pretrained=True,feature_extract=True)
    print(model)

if __name__ == "__main__":
    main()