# define data loader functions here


def load_file(path: str):
    """
    reads a text file and returns the text
    """
    with open(path, 'r') as file:
        corpus = file.read()

    return corpus