def debug_tensor(input : torch.tensor) -> str:
    """
    input : tensor object which can be allocated using torch or numpy
    """
    print("tensor is: ",input)
    print("tensor's shape is: ", input.shape)
    print("tensor's dimension is ", input.ndim)
    print("tensor's data type is: ", input.dtype)

def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    elif hasattr(obj, "to"):
        return obj.to(device)
    else:
        return obj

