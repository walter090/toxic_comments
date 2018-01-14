import os

import pandas as pd
import tensorflow as tf
from nltk.tokenize import RegexpTokenizer


def file_read_op(file_names, batch_size, num_epochs):
    """Read csv files in batch

    Args:
        file_names: List of file names
        batch_size: Int, batch size
        num_epochs: Int, number of epochs

    Returns:
        comment_text_batch: A tensor for a batch of comment text
        toxicity_batch: A tensor of a batch of one hot encoded toxicity
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


def tokenize_comments(file_dir, file_name, chunk_size, saveto='preprocessed'):
    """Tokenize the comment texts and remove the punctuations in the csv file.
    In case of a large file, process the file in chunks and append
    the chunks to new file.

    Args:
        file_dir: String directory of the file
        file_name: String file name of the original csv file
        chunk_size: Size of each chunk
        saveto: Directory to save the new file to

    Returns:
        None
    """
    df_chunk = pd.read_csv(os.path.join(file_dir, file_name), chunksize=chunk_size)
    tokenizer = RegexpTokenizer(r'\w+')
    saveto = os.path.join(file_dir, saveto)

    if not os.path.exists(saveto):
        os.makedirs(saveto)

    for index, chunk in enumerate(df_chunk):
        for row, entry in chunk.iterrows():
            chunk.at[row, 'comment_text'] = ' '.join(tokenizer.tokenize(entry['comment_text']))

        if index == 0:
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False

        chunk.to_csv(os.path.join(saveto, file_name), index=False, mode=mode, header=header)
