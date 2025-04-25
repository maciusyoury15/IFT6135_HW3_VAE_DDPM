import torch
from easydict import EasyDict
import os 

args = {
    "image_channels": 1,  # Number of channels in the image. 3 for RGB.
    "image_size": 32,  # Image size
    "n_steps": 1000,  # Number of time steps T
    "nb_save": 5,  # Number of images to save
    "batch_size": 256,  # Batch size
    "n_samples": 16,  # Number of samples to generate
    "learning_rate": 2e-4,  # Learning rate
    "epochs": 20,  # Number of training epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Device
    "fp16_precision": True,
    "show_every_n_epochs": 2,
    "MODEL_PATH": os.getcwd() + "/eps_model.pkl",
    "save_every_n_epochs": 2
}
args = EasyDict(args)