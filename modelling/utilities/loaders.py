# define data loader functions here
import pickle

def load_file(path: str):
    """
    reads a text file and returns the text
    """
    with open(path, 'r', encoding='utf-8') as file:
        corpus = file.read()

    return corpus

def load_lookup_table(path: str):
    """
    reads a text file containing a list and returns
    this
    """

    with open(path, 'rb') as file:
        char_to_idx = pickle.load(file)

    return char_to_idx

def save_lookup_table(path: str, vocab: list):
    """
    opposite of load_lookup_table()
    """

    with open(path, 'wb') as file:
        pickle.dump(vocab, file)