import os
import pickle
import string
from collections import defaultdict
from itertools import repeat
from multiprocessing import cpu_count, Pool

import nltk
import pandas as pd


def tokenize_comments(file_dir, file_name, chunk_size=20000,
                      new_dir='tokenized', lower_case=True):
    """Tokenize the comment texts and remove the punctuations in the csv file.
    In case of a large file, process the file in chunks and append
    the chunks to new file.

    Args:
        file_dir: string, directory of the file.
        file_name: string, file name of the original csv file.
        chunk_size: int, size of each chunk.
        new_dir: dict, directory to save the new file to.
        lower_case: boolean, set True to convert all words to lower case.

    Returns:
        None
    """
    df_chunk = pd.read_csv(os.path.join(file_dir, file_name), chunksize=chunk_size)
    new_dir = os.path.join(file_dir, new_dir)

    punctuations = list(string.punctuation)
    punctuations += ['``', "''"]

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for index, chunk in enumerate(df_chunk):
        print('Tokenizing chunk {}'.format(index), end='...')
        for row, entry in chunk.iterrows():
            word_list = nltk.word_tokenize(entry['comment_text'])
            word_list = [word if not lower_case else word.lower() for word in word_list if
                         word not in punctuations]
            chunk.at[row, 'comment_text'] = ' '.join(word_list)

        if index == 0:
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False

        chunk.to_csv(os.path.join(new_dir, file_name), index=False, mode=mode, header=header)
    print('Tokenization complete.')


def add_padding(file_dir, file_name, new_file=False,
                new_dir='padded', max_length=60):
    """Add padding or cut off comments to make sure all the comments have the same length.

    Args:
        file_dir: String, directory of the target csv file.
        file_name: String, name of the target csv file.
        new_dir: String, new directory to save modified csv file.
        max_length: Int, the length of comment should pad to.
        new_file: Boolean, set True to save as a new file in the specified directory,
            the operation will be performed in place otherwise.

    Returns:
        None
    """

    def pad(comment, pad_to, padword='<pad>'):
        """This function does the actual padding on the comment string.

        Args:
            comment: String, comment content.
            pad_to: Int, the length to pad the comment to.
            padword: String, a fake word to pad empty spaces in the comment.

        Returns:
            padded_comment: String, padded comment.
        """
        comment_list = comment.split(' ')
        short = pad_to - len(comment_list)

        if short > 0:
            comment_list += [padword] * short
        else:
            comment_list = comment_list[: pad_to]

        return ' '.join(comment_list)

    df = pd.read_csv(os.path.join(file_dir, file_name))
    df['comment_text'] = df['comment_text'].apply(lambda comment: pad(comment, max_length))

    # Save as new file or overwrite
    if new_file:
        os.mkdir(os.path.join(file_dir, new_dir))
        save_to = os.path.join(file_dir, new_dir, file_name)
    else:
        save_to = os.path.join(file_dir, file_name)

    df.to_csv(save_to, index=False)


def count_occurrences(file_name, chunk_size=20000, padword='<pad>'):
    """ Count occurrences of words in dataset.
    Sort all uncommon words as unknown to teach the model to deal with unseen words
    in the test set.

    Args:
        file_name: String, csv file name.
        chunk_size: Int, size of each chunk.
        padword: String, padding word used in the dataset.

    Returns:
        word_count: dict, Occurrences of each word.
    """
    df_chunks = pd.read_csv(file_name, chunksize=chunk_size)
    word_count = defaultdict(lambda: 0)

    for chunk in df_chunks:
        for _, entry in chunk.iterrows():
            comment_text = entry['comment_text'].split(' ')
            for word in comment_text:
                if word_count != padword:
                    word_count[word] += 1

    return dict(word_count)


def build_vocab(word_count, threshold=3, padword='<pad>',
                unknown='<unk>', modify=False, file_dir=None,
                file_name=None, new_dir=None, chunk_size=20000,
                uncommon_limit=500, pickle_dir=None):
    """Build a vocabulary based on words that appear in the training set.
    Words with number of occurrences below the threshold is sorted as unknown,
    this teaches the model the handel unseen words in the testing set.

    Args:
        word_count: dict, dictionary that maps a word to its number of occurrences.
        threshold: int, words with number of occurrences below this threshold is
            considered uncommon and not added to the vocabulary.
        padword: string, string value used as a padding word.
        unknown: string, string value to designate unknown words.
        modify: int, set True to modify csv file (replace less common words with
            unknown tag).
        file_dir: string, dir of base csv file.
        file_name: string, file name of base csv file.
        new_dir: string, set the argument only if you want to save the modified csv as
             a new file to a sub dir.
        chunk_size: int, size of each chunk when reading csv.
        uncommon_limit: int, size limit of the uncommon word list.
        pickle_dir: string, name of directory to store pickeled lookup dictionaries.
            the dictionary is not saved if a value is not provided. Defaults None.

    Returns:
        vocab: dict, vocabulary mapping.
        reverse_vocab: dict, reversed vocabulary mapping.
        or None if mode 2 is selected.
    """
    vocab = {unknown: -2, padword: -1}
    uncommon = []

    for index, (word, occurrences) in enumerate(word_count.items()):
        if occurrences > threshold and word != padword:
            vocab[word] = index
        elif len(uncommon) < uncommon_limit:
            uncommon.append(word)

    # Create reverse mapping vocabulary.
    reverse_vocab = {id_: word for word, id_ in vocab.items()}

    new_dir = os.path.join(file_dir, new_dir) if new_dir else file_dir
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    if modify:
        if not file_dir and not file_name:
            raise ValueError('Arguments file_dir and file_name are required.')

        df_chunks = pd.read_csv(os.path.join(file_dir, file_name), chunksize=chunk_size)
        processes = cpu_count()

        for index, chunk in enumerate(df_chunks):
            print('Processing chunk {}'.format(index), end='...')

            with Pool(processes) as pool:
                step = chunk_size // processes
                chunk_splits = [chunk.iloc[i * step: step * (i + 1)] for i in range(processes)]
                results = pool.starmap(_find_replace, zip(chunk_splits, repeat(uncommon), repeat(unknown)))
                chunk = pd.concat(results)

            if index == 0:
                mode = 'w'
                header = True
            else:
                mode = 'a'
                header = False
            chunk.to_csv(os.path.join(new_dir, file_name), index=False, mode=mode, header=header)

        print('Complete')

    if pickle_dir:
        with open(os.path.join(pickle_dir, 'vocabulary.pickle'), 'wb') as saver:
            pickle.dump((vocab, reverse_vocab), saver, protocol=pickle.HIGHEST_PROTOCOL)

    return vocab, reverse_vocab


def _find_replace(df, uncommon, unknown):
    """Helper function for building vocabulary
    """
    for row, entry in df.iterrows():
        comment_text = entry['comment_text'].split(' ')
        comment_text = [word if word not in uncommon else unknown for word in comment_text]
        df.at[row, 'comment_text'] = ' '.join(comment_text)
    return df


def translate(file_dir, file_name, vocabulary,
              new_dir=None, chunk_size=40000, word_to_id=True,
              unknown='<unk>', max_length=60):
    """Translate text in csv file either from word to id or id to word.

    Args:
        file_dir: string, directory where the csv file is found.
        file_name: string, name of the csv file.
        vocabulary: string or tuple, vocabulary look up table. If this model is
            provided as a string, it is location the pickle file is stored; otherwise,
            it is a tuple the contains the vocabulary and the reverse lookup.
        new_dir: string, directory to save the modified csv file. If this argument is
            not provided the csv file is changed in place.
        chunk_size: int, size of each chunk.
        word_to_id: boolean, set to False to translate from id to string word.
        unknown: string, designator for unseen words.
        max_length: int, max length of each comment.

    Returns:
        None.
    """
    df_chunks = pd.read_csv(os.path.join(file_dir, file_name), chunksize=chunk_size)
    new_dir = os.path.join(file_dir, new_dir) if new_dir else file_dir
    processes = cpu_count()

    if word_to_id.__class__.__name__ == 'str':
        with open(vocabulary, 'rb') as loader:
            vocab = pickle.load(loader)
    else:
        vocab = vocabulary

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for index, chunk in enumerate(df_chunks):
        print('Translating chunk {}'.format(index), end='...')

        for i in range(max_length):
            chunk['v_{}'.format(i)] = -2
            chunk['v_{}'.format(i)].astype(int, copy=False)

        with Pool(processes) as pool:
            step = chunk_size // processes
            chunk_splits = [chunk.iloc[i * step: step * (i + 1)] for i in range(processes)]
            results = pool.starmap(_translate_comment, zip(chunk_splits, repeat(word_to_id),
                                                           repeat(vocab), repeat(unknown)))
            chunk = pd.concat(results)

        if index == 0:
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False
        chunk.to_csv(os.path.join(new_dir, file_name), index=False, mode=mode, header=header)

    print('Complete')


def _translate_comment(df, word_to_id, vocab,
                       unknown):
    """Helper function for translating csv files
    """
    translation_table = vocab[0] if word_to_id else vocab[1]
    for row, entry in df.iterrows():
        comment_text = entry['comment_text'].split(' ')

        for index, word in enumerate(comment_text):
            try:
                translated_word = translation_table[word]
            except KeyError:
                translated_word = translation_table[unknown]

            df.at[row, 'v_{}'.format(index)] = translated_word

    return df
