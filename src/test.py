import seaborn, logging, matplotlib, pandas, torch, torchvision, torchaudio, omegaconf
from loggers import create_python_logger
from pathlib import Path

pylogger = create_python_logger(__name__)
Path(Path.cwd(), "logs").mkdir(parents=True, exist_ok=True)
pylogger.info("complete!")
