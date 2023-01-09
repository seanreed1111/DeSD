# import seaborn, logging, matplotlib, pandas, torch, torchvision, torchaudio, cv2, mlflow, torchmetrics, nibabel
from loggers import create_python_logger
from pathlib import Path
from models.res3d import res3d50
import torch, torchvision
from omegaconf import OmegaConf

from torch import nn

pylogger = create_python_logger(__name__)

if __name__ == "__main__":
    args = OmegaConf.from_cli()
    # pylogger.debug(f"torchvision model list \n{torchvision.models}")
    model = res3d50()
    state_dict = model.state_dict()
    pylogger.debug(f"model \n {model}")
    pylogger.debug(f"state dict \n {state_dict}")
    # model.load_state_dict(torch.load(args.load_path))
    # pylogger.debug("model loaded successfully")
    # pylogger.debug(f"{model}")
