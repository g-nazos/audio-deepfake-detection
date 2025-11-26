import os


def get_file_path(file, dataset_pathing, label):
    """
    This function gets the system path of the input audio file path.
    Input:
    path = Input audio file path (relative to the current working directory)
    returns:
    System path of the input audio file path if it exists, otherwise None
    """
    try:
        cwd = os.getcwd()
    except Exception as e:
        print(f"Error getting current working directory: {e}")
        return None
    if label == "spoof":
        type = "fake"
        file_path = os.path.join(cwd, dataset_pathing, type, file)
    elif label == "bona-fide":
        type = "real"
        file_path = os.path.join(cwd, dataset_pathing, type, file)
    else:
        raise FileNotFoundError(f"Label given doesn't match!")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist")
    return file_path
