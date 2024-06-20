import logging
import os

import torch


def get_device(use_gpu_acceleration=True):
    """

    Parameters
    ----------
    use_gpu_acceleration
    Returns
    ------
    """
    device_label = "cpu"

    if use_gpu_acceleration:
        if torch.backends.mps.is_available():
            logging.info("Using device: <mps>")
            device_label = torch.device("mps")
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        else:
            logging.info("No apple silicone supported")
            if torch.cuda.is_available():
                logging.info("Using <torch cuda> ")
                device_label = torch.device("cuda:0")
                torch.backends.cudnn.benchmark = True

    return device_label
