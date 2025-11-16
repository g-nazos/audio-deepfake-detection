import os


def get_file_path(path):
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
    file_path = os.path.join(cwd, path)
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return None
    return file_path
