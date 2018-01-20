import os
import string

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from itertools import repeat
from multiprocessing import cpu_count, Pool


def file_read_op(file_names, batch_size, num_epochs):
    """Read csv files in batch

    Args:
        file_names: List of file names.
        batch_size: Int, batch size.
        num_epochs: Int, number of epochs.

    Returns:
        comment_text_batch: A tensor for a batch of comment text.
        toxicity_batch: A tensor of a batch of one hot encoded toxicity.
    """
    reader = tf.TextLineReader(skip_header_lines=1)
    queue = tf.train.string_input_producer(file_names,
                                           num_epochs=num_epochs,
                                           shuffle=True)

    _, value = reader.read(queue)
    record_defaults = [[''], [''], [0], [0], [0], [0], [0], [0]]
    cols = tf.decode_csv(value, record_defaults=record_defaults)
    comment_text = cols[1]  # Skip id column
    toxicity = tf.stack(cols[2:])

    min_after_dequeue = 10000
    capacity = min_after_dequeue + 4 * batch_size
    comment_text_batch, toxicity_batch = tf.train.shuffle_batch(
        [comment_text, toxicity], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    return comment_text_batch, toxicity_batch


def tokenize_comments(file_dir, file_name, chunk_size=20000, new_dir='tokenized'):
    """Tokenize the comment texts and remove the punctuations in the csv file.
    In case of a large file, process the file in chunks and append
    the chunks to new file.

    Args:
        file_dir: String directory of the file.
        file_name: String file name of the original csv file.
        chunk_size: Size of each chunk.
        new_dir: Directory to save the new file to.

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
            word_list = [word for word in word_list if word not in punctuations]
            chunk.at[row, 'comment_text'] = ' '.join(word_list)

        if index == 0:
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False

        chunk.to_csv(os.path.join(new_dir, file_name), index=False, mode=mode, header=header)
    print('Tokenization complete.')


def add_padding(file_dir, file_name, new_file=False, new_dir='padded', max_length=60):
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


def build_vocab(word_count, threshold=3, padword='<pad>', unknown='<unk>', modify=False,
                file_dir=None, file_name=None, new_dir=None, chunk_size=20000):
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

    Returns:
        vocab: dict, vocabulary mapping.
        reverse_vocab: dict, reversed vocabulary mapping.
        or None if mode 2 is selected.
    """
    vocab = {unknown: -2, padword: -1}
    uncommon = []

    for index, (word, occurrences) in enumerate(word_count.items()):
        if occurrences > threshold:
            vocab[word] = index
        else:
            uncommon.append(word)
    # Create reverse mapping vocabulary.
    reverse_vocab = {id_: word for word, id_ in vocab.items()}

    if modify:
        if not file_dir and not file_name:
            raise ValueError('Arguments file_dir and file_name are required.')

        new_dir = os.path.join(file_dir, new_dir) if new_dir else file_dir
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)

        df_chunks = pd.read_csv(os.path.join(file_dir, file_name), chunksize=chunk_size)
        for index, chunk in enumerate(df_chunks):
            print('Processing chunk {}'.format(index), end='...')

            processes = cpu_count()
            with Pool(processes) as pool:
                chunk_splits = np.split(chunk, processes)
                pool.starmap(_find_replace, zip(chunk_splits, repeat(uncommon), repeat(unknown)))

            if index == 0:
                mode = 'w'
                header = True
            else:
                mode = 'a'
                header = False
            chunk.to_csv(os.path.join(new_dir, file_name), index=False, mode=mode, header=header)

        print('Complete')

    return vocab, reverse_vocab


def _find_replace(df, uncommon, unknown):
    """Helper function for building vocabulary

    """
    for row, entry in df.iterrows():
        comment_text = entry['comment_text'].split(' ')
        comment_text = [word if word not in uncommon else unknown for word in comment_text]
        df.at[row, 'comment_text'] = ' '.join(comment_text)
