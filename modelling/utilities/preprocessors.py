# define data preprocessor functions here
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

def map_value_to_index(unique_ids, n_unique_ids, start, inverted=False):
    """
    returns a dictionary mapping each unique value to an integer. 
    This is akin to generating a word to index dictionary where each
    unique word based on their freqeuncy will be mapped from indeces
    1 to |V|.

    e.g. >>> start = 0
    >>> val_to_index = dict(zip(ids, list(range(start, n_ids + start))))
    >>> val_to_index
    {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    >>>
    >>> start = 10
    >>> val_to_index = dict(zip(ids, list(range(start, n_ids + start))))
    >>> val_to_index
    {1: 10, 2: 11, 3: 12, 4: 13, 5: 14}

    args:
        unique_user_ids - an array/vector/set of all unique user id's from
        perhaps a ratings dataset
    """

    return dict(zip(unique_ids, list(range(start, n_unique_ids + start)))) \
    if inverted is False else dict(zip(list(range(start, n_unique_ids + start)), unique_ids))

def init_sequences(corpus: str, char_to_idx: dict, T_x: int):
    """
    turns our raw preprocessed corpus to a list of
    sequences each of which has a defined max length T_x.
    Note T_x must be less than or e.g. if there are 150000
    characters in the corpus then T_x must only be 149999
    this is to prevent an "index out of range" error when
    generating a label/target y character

    e.g. "first step down.
    i was with jordan peterson.
    meaning is to be drawn from suffering."

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
        in_seq = [char_to_idx[ch] for ch in corpus[i: i + T_x]]
        out_seq = char_to_idx[corpus[i + T_x]]

        # append input and output sequences
        in_seqs.append(in_seq)
        out_seqs.append(out_seq)
    
    return np.array(in_seqs), np.array(out_seqs)

    