import os
import pickle
import string
from collections import defaultdict
from itertools import repeat
from multiprocessing import cpu_count, Pool

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords


def tokenize_comments(file_dir, file_name, chunk_size=20000,
                      new_dir=None, new_name='tokenized.csv', lower_case=True,
                      keep_stopwords=False):
    """Tokenize the comment texts and remove the punctuations in the csv file.
    In case of a large file, process the file in chunks and append
    the chunks to new file.

    Args:
        new_name: string, name for new file.
        file_dir: string, directory of the file.
        file_name: string, file name of the original csv file.
        chunk_size: int, size of each chunk.
        new_dir: dict, directory to save the new file to.
        lower_case: boolean, set True to convert all words to lower case.
        keep_stopwords: boolean, set True to keep stopwords in document.

    Returns:
        New file location
    """
    df_chunk = pd.read_csv(os.path.join(file_dir, file_name), chunksize=chunk_size)
    new_dir = os.path.join(file_dir, new_dir) if new_dir else file_dir

    punctuations = list(string.punctuation)
    punctuations += ['``', "''"]

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for index, chunk in enumerate(df_chunk):
        print('Tokenizing chunk {}'.format(index), end='...')
        for row, entry in chunk.iterrows():
            try:
                word_list = nltk.word_tokenize(entry['comment_text'])
            except TypeError:
                continue
            word_list = [word if not lower_case else word.lower() for word in word_list if
                         word not in punctuations and (word not in stopwords and not keep_stopwords)]
            chunk.at[row, 'comment_text'] = ' '.join(word_list)

        if index == 0:
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False

        chunk.to_csv(os.path.join(new_dir, new_name), index=False, mode=mode, header=header)
    print('Tokenization complete.')
    return os.path.join(file_dir, new_name)


def add_padding(file_dir, file_name, new_dir=None,
                max_length=60, new_name='padded.csv'):
    """Add padding or cut off comments to make sure all the comments have the same length.

    Args:
        file_dir: String, directory of the target csv file.
        file_name: String, name of the target csv file.
        new_dir: String, new directory to save modified csv file.
        max_length: Int, the length of comment should pad to.
        new_name: string, name for new saved file.

    Returns:
        New file location
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
        comment = str(comment)
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
    if new_dir:
        os.mkdir(os.path.join(file_dir, new_dir))
        save_to = os.path.join(file_dir, new_dir, new_name)
    else:
        save_to = os.path.join(file_dir, new_name)

    df.to_csv(save_to, index=False)
    return save_to


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
                if word != padword:
                    word_count[word] += 1

    return dict(word_count)


def build_vocab(word_count, threshold=5, padword='<pad>',
                unknown='<unk>', modify=False, file_dir=None,
                file_name=None, new_dir=None, new_name='replaced.csv',
                chunk_size=20000, uncommon_limit=500, pickle_dir=None,
                tsv_dir=None):
    """Build a vocabulary based on words that appear in the training set.
    Words with number of occurrences below the threshold is sorted as unknown,
    this teaches the model the handel unseen words in the testing set.

    Args:
        new_name: string, name for new saved file.
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
        tsv_dir: string, name of directory to store tsv look up.

    Returns:
        vocab: dict, vocabulary mapping.
        reverse_vocab: dict, reversed vocabulary mapping.
        or None if mode 2 is selected.
    """
    vocab = {unknown: 0, padword: 1}
    uncommon = []

    index = 2
    for word, occurrences in word_count.items():
        if occurrences > threshold and word != padword and word != unknown:
            vocab[word] = index
            index += 1
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
            chunk.to_csv(os.path.join(new_dir, new_name), index=False, mode=mode, header=header)

        print('Complete')

    if pickle_dir:
        if not os.path.exists(pickle_dir):
            os.makedirs(pickle_dir)

        with open(os.path.join(pickle_dir, 'vocabulary.pickle'), 'wb') as saver:
            pickle.dump((vocab, reverse_vocab), saver, protocol=pickle.HIGHEST_PROTOCOL)

    if tsv_dir:
        if not os.path.exists(tsv_dir):
            os.makedirs(tsv_dir)

        with open(os.path.join(tsv_dir, 'metadata.tsv'), 'w') as meta_saver:
            meta_saver.write('ID\tWord\n')
            for id_, word in reverse_vocab.items():
                meta_saver.write('{}\t{}\n'.format(id_, word))

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
              new_dir=None, new_name='translated.csv', chunk_size=40000,
              word_to_id=True, unknown='<unk>', max_length=60,
              translate_mode='document', ngram_name='ngram.csv', window=3):
    """Translate text in csv file either from word to id or id to word.

    Args:
        translate_mode: string, mode of translation.
        ngram_name: string, name for ngram file.
        window: int, window size for skip gram.
        new_name: string, name for new file.
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

    if vocabulary.__class__.__name__ == 'str':
        ext = vocabulary.split('.')[-1]
        if ext == 'pickle':
            with open(vocabulary, 'rb') as loader:
                vocab = pickle.load(loader)
        elif ext == 'tsv':
            vocab = [{}, {}]
            with open(vocabulary) as file:
                for line in file:
                    id_, word = line.split()
                    try:
                        vocab[0][word] = int(id_)
                        vocab[1][int(id_)] = word
                    except ValueError:
                        continue
    else:
        vocab = vocabulary

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for index, chunk in enumerate(df_chunks):
        print('Translating chunk {}'.format(index), end='...')

        if index == 0:
            mode = 'w'
            header = True
        else:
            mode = 'a'
            header = False

        step = chunk_size // processes

        if translate_mode == 'list':
            translate_chunk = chunk
            for i in range(max_length):
                translate_chunk['v_{}'.format(i)] = 0  # 0 for unknown
                translate_chunk['v_{}'.format(i)].astype(int, copy=False)

            with Pool(processes) as pool:
                chunk_splits = [translate_chunk.iloc[i * step: step * (i + 1)] for i in range(processes)]
                results = pool.starmap(_translate_comment, zip(chunk_splits, repeat(word_to_id),
                                                               repeat(vocab), repeat(unknown)))
                translate_chunk = pd.concat(results)
            translate_chunk.to_csv(os.path.join(new_dir, new_name), index=False, mode=mode, header=header)

        elif translate_mode == 'ngram':
            ngram_chunk = chunk
            with Pool(processes) as pool:
                splits = [ngram_chunk.iloc[i * step: step * (i + 1)] for i in range(processes)]
                results = pool.starmap(_create_ngram, zip(splits, repeat(window)))
                new_ngram = pd.concat(results)
            new_ngram.to_csv(os.path.join(new_dir, ngram_name), index=False, mode=mode, header=header)

        elif translate_mode == 'document':
            str_chunk = chunk
            with Pool(processes) as pool:
                splits = [str_chunk.iloc[i * step: step * (i + 1)] for i in range(processes)]
                results = pool.starmap(_translate_comment_str, zip(splits, repeat(word_to_id),
                                                                   repeat(vocab), repeat(unknown)))
                str_chunk = pd.concat(results)
            str_chunk.to_csv(os.path.join(new_dir, new_name), index=False, mode=mode, header=header)

    print('Complete')


def _create_ngram(df, window):
    """Helper function for creating ngram
    """
    ngram_df = pd.DataFrame(columns=['target', 'context'])
    either_side = (window - 1) // 2

    for _, entry in df.iterrows():
        comment_text = entry['comment_text'].split(' ')

        for index in range(either_side, len(comment_text) - either_side):
            for pos in range(1, either_side + 1):
                ngram_df.loc[len(ngram_df)] = [comment_text[index], comment_text[index - pos]]
                ngram_df.loc[len(ngram_df)] = [comment_text[index], comment_text[index + pos]]
    return ngram_df


def _translate_comment_str(df, word_to_id, vocab,
                           unknown):
    translation_table = vocab[0] if word_to_id else vocab[1]
    for row, entry in df.iterrows():
        comment_text = entry['comment_text'].split(' ')
        translated = []
        for index, word in enumerate(comment_text):
            try:
                translated_word = str(translation_table[word])
            except KeyError:
                translated_word = str(translation_table[unknown])
            translated.append(translated_word)
        df.at[row, 'comment_text'] = ' '.join(translated)
    return df


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


def split_data(file_dir, file_name, test_size=0.2):
    df = pd.read_csv(os.path.join(file_dir, file_name))
    df_test = df.sample(frac=test_size)
    df_train = df.drop(df_test.index)

    df_train.to_csv(os.path.join(file_dir, 'train_split.csv'), index=False)
    df_test.to_csv(os.path.join(file_dir, 'test_split.csv'), index=False)


def build_vocab_from_file(vec_file, pad='<pad>', unknown='<unk>',
                          limit=30000):
    """ Extract vocabulary and embeddings from pre trained embedding file.

    Args:
        vec_file: string, name of embedding file.
        pad: string, pad word token.
        unknown: string, unknown word token
        limit: int, upper limit of vocab.

    Returns:
        word2id: dict, string word to id mapping.
        embeddings: list, list of word vectors.
    """
    word2id = {pad: 0, unknown: 1}
    embeddings = []

    with open(vec_file) as file:
        for index, entry in enumerate(file):
            values = entry.split()
            word = values[0]
            weights = np.asarray(values[1:], dtype=np.float32)

            word2id[word] = index + 2
            embeddings.append(weights)

            if index + 1 == limit:
                break

    embedding_size = len(embeddings[0])
    # Random for unknown
    embeddings.insert(0, np.random.randn(embedding_size))
    # Random for padding, padding will be masked during lookup.
    embeddings.insert(0, np.random.randn(embedding_size))

    embeddings = np.asarray(embeddings, dtype=np.float32)

    return word2id, embeddings
