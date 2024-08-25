import torch

def save_model(model, path):
    """
    Save the model's state_dict to a file.

    Parameters:
    model (torch.nn.Module): The model to be saved.
    path (str): The path where the model will be saved.
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model_class, path, *args, **kwargs):
    """
    Load the model's state_dict from a file.

    Parameters:
    model_class (torch.nn.Module): The class of the model to be loaded.
    path (str): The path from where the model will be loaded.
    *args: Positional arguments to instantiate the model.
    **kwargs: Keyword arguments to instantiate the model.

    Returns:
    model (torch.nn.Module): The model with loaded weights.
    """
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {path}")
    return model

