import torch
import logging
import sys
import sacred

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


def sanitize_sacred_arguments(args):
    """
    This function goes through and sanitizes the arguments to native types.
    Lists and dictionaries passed through Sacred automatically become
    ReadOnlyLists and ReadOnlyDicts. This function will go through and
    recursively change them to native lists and dicts.
    `args` can be a single token, a list of items, or a dictionary of items.
    The return type will be a native token, list, or dictionary.
    """
    if isinstance(args, list):  # Captures ReadOnlyLists
        return [
            sanitize_sacred_arguments(item) for item in args
        ]
    elif isinstance(args, dict):  # Captures ReadOnlyDicts
        return {
            str(key) : sanitize_sacred_arguments(val) \
            for key, val in args.items()
        }
    else:  # Return the single token as-is
        return args


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
