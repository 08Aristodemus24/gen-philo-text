# define data preprocessor functions here
import tensorflow as tf

import re
import numpy as np

def remove_contractions(text_string: str):
    """
    removes contractions and replace them e.g. don't becomes
    do not and so on
    """

    text_string = re.sub(r"don't", "do not ", text_string)
    text_string = re.sub(r"didn't", "did not ", text_string)
    text_string = re.sub(r"aren't", "are not ", text_string)
    text_string = re.sub(r"weren't", "were not", text_string)
    text_string = re.sub(r"isn't", "is not ", text_string)
    text_string = re.sub(r"can't", "cannot ", text_string)
    text_string = re.sub(r"doesn't", "does not ", text_string)
    text_string = re.sub(r"shouldn't", "should not ", text_string)
    text_string = re.sub(r"couldn't", "could not ", text_string)
    text_string = re.sub(r"mustn't", "must not ", text_string)
    text_string = re.sub(r"wouldn't", "would not ", text_string)

    text_string = re.sub(r"what's", "what is ", text_string)
    text_string = re.sub(r"that's", "that is ", text_string)
    text_string = re.sub(r"he's", "he is ", text_string)
    text_string = re.sub(r"she's", "she is ", text_string)
    text_string = re.sub(r"it's", "it is ", text_string)
    text_string = re.sub(r"that's", "that is ", text_string)

    text_string = re.sub(r"could've", "could have ", text_string)
    text_string = re.sub(r"would've", "would have ", text_string)
    text_string = re.sub(r"should've", "should have ", text_string)
    text_string = re.sub(r"must've", "must have ", text_string)
    text_string = re.sub(r"i've", "i have ", text_string)
    text_string = re.sub(r"we've", "we have ", text_string)

    text_string = re.sub(r"you're", "you are ", text_string)
    text_string = re.sub(r"they're", "they are ", text_string)
    text_string = re.sub(r"we're", "we are ", text_string)

    text_string = re.sub(r"you'd", "you would ", text_string)
    text_string = re.sub(r"they'd", "they would ", text_string)
    text_string = re.sub(r"she'd", "she would ", text_string)
    text_string = re.sub(r"he'd", "he would ", text_string)
    text_string = re.sub(r"it'd", "it would ", text_string)
    text_string = re.sub(r"we'd", "we would ", text_string)

    text_string = re.sub(r"you'll", "you will ", text_string)
    text_string = re.sub(r"they'll", "they will ", text_string)
    text_string = re.sub(r"she'll", "she will ", text_string)
    text_string = re.sub(r"he'll", "he will ", text_string)
    text_string = re.sub(r"it'll", "it will ", text_string)
    text_string = re.sub(r"we'll", "we will ", text_string)

    text_string = re.sub(r"\n't", " not ", text_string) #
    text_string = re.sub(r"\'s", " ", text_string) 
    text_string = re.sub(r"\'ve", " have ", text_string) #
    text_string = re.sub(r"\'re", " are ", text_string) #
    text_string = re.sub(r"\'d", " would ", text_string) #
    text_string = re.sub(r"\'ll", " will ", text_string) # 
    
    text_string = re.sub(r"i'm", "i am ", text_string)

    return text_string

def preprocess(text_string: str):
    """
    will follow a pipeline of removing and replacing unnecessary
    characters from the given corpus of text
    """

    # turn sentences to lowercase
    temp = text_string.lower()

    # replace chars '“', '”' with """ instead
    temp = temp.replace('“', '"')
    temp = temp.replace('”', '"')

    # replace chars '‘', '’' with "'" instead
    temp = temp.replace('‘', "'")
    temp = temp.replace('’', "'")

    # replace medium hyphen '–' with longer hyphen '—'
    temp = temp.replace('–', '—')

    # replace 3 consecutive '.' with  '…' instead
    temp = re.sub(r"[.]{3,}", "…", temp)

    # following substitutions are for words with contractions e.g. don't -> do nots
    temp = remove_contractions(temp)

    # remove whitespaces
    temp = temp.strip()

    return temp

def get_chars(corpus: str):
    """
    returns a list of all unique characters found
    in given corpus
    """
    chars = sorted(list(set(corpus)))

    return chars

# def map_value_to_index(unique_ids, n_unique_ids, start, inverted=False):
#     """
#     returns a dictionary mapping each unique value to an integer. 
#     This is akin to generating a word to index dictionary where each
#     unique word based on their freqeuncy will be mapped from indeces
#     1 to |V|.

#     e.g. >>> start = 0
#     >>> val_to_index = dict(zip(ids, list(range(start, n_ids + start))))
#     >>> val_to_index
#     {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
#     >>>
#     >>> start = 10
#     >>> val_to_index = dict(zip(ids, list(range(start, n_ids + start))))
#     >>> val_to_index
#     {1: 10, 2: 11, 3: 12, 4: 13, 5: 14}

#     args:
#         unique_user_ids - an array/vector/set of all unique user id's from
#         perhaps a ratings dataset
#     """

#     return dict(zip(unique_ids, list(range(start, n_unique_ids + start)))) \
#     if inverted is False else dict(zip(list(range(start, n_unique_ids + start)), unique_ids))

def map_value_to_index(unique_tokens: list, inverted=False):
    """
    returns a lookup table mapping each unique value to an integer. 
    This is akin to generating a word to index dictionary where each
    unique word based on their freqeuncy will be mapped from indeces
    1 to |V|.

    args:
        unique_tokens - 
        inverted - 
    """
    char_to_idx = tf.keras.layers.StringLookup(vocabulary=unique_tokens, mask_token=None)
    idx_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_idx.get_vocabulary(), invert=True, mask_token=None)

    return char_to_idx if inverted == False else idx_to_char

def init_sequences_a(corpus: str, char_to_idx: dict, T_x: int):
    """
    turns our raw preprocessed corpus to a list of
    sequences each of which has a defined max length T_x.
    Note T_x must be less than or e.g. if there are 150000
    characters in the corpus then T_x must only be 149999
    this is to prevent an "index out of range" error when
    generating a label/target y character.

    e.g. "first step down.
    i was with jordan peterson.
    meaning is to be drawn from suffering."

    with T_x of 5 becomes...

    X = [
        ['f', 'i', 'r', 's', 't],
        ['i', 'r', 's', 't', ' '],
        ...
    ]

    Y = [
        ' ',
        's',
        ...
    ]

    of course this is not entirely the dataset we would get
    after because each character token as we know will be
    converted to its respective id/index. This merely to understand
    what our the alternative model a will take in as input.

    args:
        corpus - 
        char_to_idx - 
        T_x - 
    """
    # get total length of corpus
    total_len = len(corpus)

    # will contain our training examples serving as 
    # our x and y values
    in_seqs = []
    out_seqs = []

    # generate pairs of input and output sequences that 
    # will serve as our training examples x and y
    for i in range(0, total_len - T_x):
        # slice corpus into input and output characters 
        # and convert each character into their respective 
        # indeces using char_to_idx mapping
        in_seq = [ch for ch in corpus[i: i + T_x]]
        out_seq = corpus[i + T_x]

        # append input and output sequences
        in_seqs.append(in_seq)
        out_seqs.append(out_seq)
    
    return char_to_idx(in_seqs), char_to_idx(out_seqs)

def init_sequences_b(corpus: str, char_to_idx: dict, T_x: int):
    """
    generates a input and target dataset by:

    1. partitioning corpus first into sequences of length T_x + 1
    2. shifting sequences by one character to the left to generate 
    output/target sequence the model needs to learn

    A sequence length of 0 will not be permitted and this
    funciton will raise an error should T_x be 0
    """

    if T_x == 0:
        raise ValueError("You have entered an unpermitted value for the number of timesteps T_x. Sequence length T_x cannot be 0. Choose a value above 0.")

    # get total length of corpus
    total_len = len(corpus)

    # will contain our training examples serving as 
    # our x and y values
    in_seqs = []
    out_seqs = []

    # generate pairs of input and output sequences that 
    # will serve as our training examples x and y
    # loop through each character and every T_x char append it to the
    for i in range(0, total_len, T_x + 1):
        # slice corpus into input and output characters 
        # and convert each character into their respective 
        # indeces using char_to_idx mapping
        partition = [ch for ch in corpus[i: i + (T_x + 1)]]
        in_seq = partition[:-1]
        out_seq = partition[1:]

        # append input and output sequences
        in_seqs.append(in_seq)
        out_seqs.append(out_seq)

    if total_len % (T_x + 1):
        # calculate number of chars missing in last training example
        n_chars_missed = T_x - len(in_seqs[-1])

        # pad with zeroes to example with less than 100 chars
        in_seqs[-1] = in_seqs[-1] + (['[UNK]'] * n_chars_missed)
        out_seqs[-1] = out_seqs[-1] + (['[UNK]'] * n_chars_missed)

    return char_to_idx(in_seqs), char_to_idx(out_seqs)

def decode_predictions(pred_ids, idx_to_char):
    """
    decodes the predictions by inference model and converts
    them into the full generated sentence itself
    """
    char_list = idx_to_char(pred_ids)
    char_list = tf.reshape(char_list, shape=(-1,)).numpy()
    joined_seq = b"".join(char_list)
    final_seq = str(joined_seq, "utf-8")

    return final_seq






    