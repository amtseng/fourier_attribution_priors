import torch
import logging
import sys

def make_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(process)d: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def place_tensor(tensor):
    """
    Places a tensor on GPU, if PyTorch sees CUDA; otherwise, the returned tensor
    remains on CPU.
    """
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def save_model(model, save_path):
    """
    Saves the given model at the given path. This saves the state of the model
    (i.e. trained layers and parameters), and the arguments used to create the
    model (i.e. a dictionary of the original arguments).
    """
    save_dict = {
        "model_state": model.state_dict(),
        "model_creation_args": model.creation_args
    }
    torch.save(save_dict, save_path)


def restore_model(model_class, load_path):
    """
    Restores a model from the given path. `model_class` must be the class for
    which the saved model was created from. This will create a model of this
    class, using the loaded creation arguments. It will then restore the learned
    parameters to the model.
    """
    load_dict = torch.load(load_path)
    model_state = load_dict["model_state"]
    model_creation_args = load_dict["model_creation_args"]
    model = model_class(**model_creation_args)
    model.load_state_dict(model_state)
    return model
