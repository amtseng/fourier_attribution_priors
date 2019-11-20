import torch
import logging
import sys
import sacred
import scipy.ndimage
import numpy as np

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


def convolution_size(
    given_size, num_layers, kernel_sizes, padding=0, strides=1, dilations=1,
    inverse=False
):
    """
    Computes the size of the convolutional output after applying several layers
    of convolution to an input of a given size. Alternatively, this can also
    compute the size of a convolutional input needed to create the given size
    for an output.
    Arguments:
        `given_size`: the size of an input sequence, or the size of a desired
            output sequence
        `num_layers`: number of convolutional layers to apply
        `kernel_sizes`: array of kernel sizes, to be applied in order; can also
            be an integer, which is the same kernel size for all layers
        `padding`: array of padding amounts, with each value being the amount of
            padding on each side of the input at each layer; can also be an
            integer, which is the same padding for all layers
        `strides`: array of stride values, with each value being the stride
            at each layer; can also be an integer, which is the same stride for
            all layers
        `dilations`: array of dilation values, with each value being the
            dilation at each layer; can also be an integer, which is the same
            dilation for all layers
        `inverse`: if True, computes the size of input needed to generate an
            output of size `given_size`
    Returns the size of the sequence after convolutional layers of these
    specifications are applied in order.
    """
    if type(kernel_sizes) is int:
        kernel_sizes = [kernel_sizes] * num_layers
    else:
        assert len(kernel_sizes) == num_layers
    if type(padding) is int:
        padding = [padding] * num_layers
    else:
        assert len(padding) == num_layers
    if type(strides) is int:
        strides = [strides] * num_layers
    else:
        assert len(strides) == num_layers
    if type(dilations) is int:
        dilations = [dilations] * num_layers
    else:
        assert len(dilations) == num_layers

    size = given_size

    if not inverse:
        for i in range(num_layers):
            size = int(
                (size + (2 * padding[i]) - (dilations[i] * (kernel_sizes[i] - 1)) \
                 - 1) / strides[i]
            ) + 1
    else:
        for i in range(num_layers - 1, -1, -1):
            size = (strides[i] * (size - 1)) - (2 * padding[i]) + \
                   (dilations[i] * (kernel_sizes[i] - 1)) + 1
    return size


def smooth_tensor_1d(input_tensor, smooth_sigma):
    """
    Smooths an input tensor along a dimension using a Gaussian filter.
    Arguments:
        `input_tensor`: a A x B tensor to smooth along the second dimension
        `smooth_sigma`: width of the Gaussian to use for smoothing; this is the
            standard deviation of the Gaussian to use, and the Gaussian will be
            truncated after 1 sigma (i.e. the smoothing window is
            1 + (2 * sigma); sigma of 0 means no smoothing
    Returns an array the same shape as the input tensor, with the dimension of
    `B` smoothed.
    """
    # Generate the kernel
    if smooth_sigma == 0:
        sigma, truncate = 1, 0
    else:
        sigma, truncate = smooth_sigma, 1
    base = np.zeros(1 + (2 * sigma))
    base[sigma] = 1  # Center of window is 1 everywhere else is 0
    kernel = scipy.ndimage.gaussian_filter(base, sigma=sigma, truncate=truncate)
    kernel = place_tensor(torch.tensor(kernel))

    # Expand the input and kernel to 3D, with channels of 1
    # Also make the kernel float-type, as the input is going to be of type float
    input_tensor = torch.unsqueeze(input_tensor, dim=1)
    kernel = torch.unsqueeze(torch.unsqueeze(kernel, dim=0), dim=1).float()

    smoothed = torch.nn.functional.conv1d(
        input_tensor, kernel, padding=sigma
    )

    return torch.squeeze(smoothed, dim=1)
