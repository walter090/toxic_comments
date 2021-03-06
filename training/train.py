import argparse
import datetime
import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import preprocess
from model.toxic_detection import ToxicityCNN
from model.rnn import ToxicityLSTM
from model.skip_gram import WordEmbedding


def train(model, verbose_freq=200, save_freq=2000,
          meta=None, log_dir=None, model_dir=None,
          metadata=None, word_vector_meta=None, word_vector_file=None,
          vocab=60000, args_config=None):
    """Function for training models

    Args:
        model: Model, object of the model to be trained.
        verbose_freq: int, frequency of update message.
        save_freq: int, step interval to save the variables.
        meta: string, path to the saved variables for continue training. Optional,
            defaults None.
        log_dir: string, directory to save tensorboard summaries.
        model_dir: string, directory to save variables.
        metadata: string, path to the metadata that maps IDs to words.
        word_vector_meta: string, path to the saved word vectors.
        word_vector_file: string, path to the pre trained word vectors. Cannot
            be used with word_vector_meta; if both arguments are provided,
            word_vector_meta will be ignored.
        vocab: int, vocabulary size.
        args_config: dict, network configuration.

    Returns:
        None
    """
    timestamp = datetime.datetime.now().isoformat('_')
    save_dir = os.path.abspath(os.path.join(os.path.curdir, 'models_and_visual', timestamp))
    log_dir = os.path.join(save_dir, 'tensorboard') if not log_dir else log_dir
    model_dir = os.path.join(save_dir, 'saved_models') if not model_dir else model_dir

    if args_config:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, 'config.tsv'), 'w') as config_saver:
            config_saver.write('Arg\tValue\n')
            for key, value in args_config.items():
                config_saver.write('{}\t{}\n'.format(key, value))

    if word_vector_file:
        word2id, vec = preprocess.build_vocab_from_file(word_vector_file, limit=vocab)
        model.provide_vector(vec)

    model_grads, model_optimization = model.optimize
    model_step = model.global_step
    model_loss = model.loss
    model_auc = model.metric
    model_embeddings = model.embeddings[0]

    tf.summary.scalar('loss', model_loss)
    tf.summary.scalar('AUC', model_auc)

    for grad_i, grad in enumerate(model_grads):
        tf.summary.histogram('grad_{}'.format(grad[0].name), grad[0])

    all_summaries = tf.summary.merge_all()

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        projector_config = projector.ProjectorConfig()
        embedding_projector = projector_config.embeddings.add()
        embedding_projector.tensor_name = model_embeddings.name
        if metadata:
            embedding_projector.metadata_path = metadata

        projector.visualize_embeddings(writer, projector_config)

        sess.run(init_op)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=7)

        if meta:
            restore_variables(meta, sess)
        if word_vector_meta and not word_vector_file:
            restore_word_vectors(word_vector_meta, sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                _, step, loss, auc, summaries = sess.run([model_optimization, model_step,
                                                          model_loss, model_auc,
                                                          all_summaries])

                writer.add_summary(summaries, global_step=step)

                cur_time = datetime.datetime.now().isoformat('_')
                if step % verbose_freq == 0:
                    print('{} - At step {}, loss {}, AUC {}'.format(cur_time, step, loss, auc))
                if step % save_freq == 0:
                    saver.save(sess, save_path=model_dir, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, save_path=model_dir)


def test(model, meta, verbose_freq=20):
    model_auc = model.metric

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        restore_variables(meta, sess)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            step = 0
            while not coord.should_stop():
                auc = sess.run(model_auc)
                if step % verbose_freq == 0:
                    print('At step {}: AUC {}'.format(step, auc))
                step += 1
        except tf.errors.OutOfRangeError:
            print('Done testing.')
        finally:
            coord.request_stop()

        coord.join(threads)


def restore_variables(meta, sess):
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=meta)


def restore_word_vectors(meta, sess):
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                      'embedding'))
    saver.restore(sess=sess, save_path=meta)


def test_cnn(csvs, meta, vocab, batch_size=516,
             num_epochs=1, embedding_size=300):
    model = ToxicityCNN(csvs=csvs, vocab_size=vocab, batch_size=batch_size,
                        num_epochs=num_epochs, embedding_size=embedding_size, num_labels=6,
                        comment_length=60, testing=True)
    test(model, meta=meta)


def test_lstm(csvs, meta, vocab,
              peepholes, bi, batch_size=4096,
              num_epochs=1, embedding_size=300, num_layers=2,
              attention=False, comment_length=None, verbose_freq=20):
    vocab += 2
    model = ToxicityLSTM(csvs=csvs, vocab_size=vocab, batch_size=batch_size,
                         num_epochs=num_epochs, embedding_size=embedding_size, num_labels=6,
                         comment_length=comment_length, testing=True, peepholes=peepholes,
                         bi=bi, num_layers=num_layers, attention=attention)
    test(model, meta=meta, verbose_freq=verbose_freq)


def train_cnn(csvs, vocab_size=18895, batch_size=512,
              num_epochs=150, embedding_size=100, num_labels=6,
              comment_length=60, verbose_freq=200, save_freq=2000,
              word_vector_meta=None, meta=None, log_dir=None,
              model_dir=None, metadata=None, vector_file=None,
              args_config=None):
    model = ToxicityCNN(csvs=csvs, batch_size=batch_size,
                        num_epochs=num_epochs, vocab_size=vocab_size,
                        embedding_size=embedding_size, num_labels=num_labels,
                        comment_length=comment_length)
    train(model=model, verbose_freq=verbose_freq, save_freq=save_freq,
          meta=meta, log_dir=log_dir, model_dir=model_dir,
          metadata=metadata, word_vector_meta=word_vector_meta, word_vector_file=vector_file,
          args_config=args_config)


def train_lstm(csvs, vocab_size=None, batch_size=256,
               num_epochs=100, num_labels=6,
               comment_length=60, verbose_freq=200, save_freq=1000,
               word_vector_meta=None, meta=None, log_dir=None,
               model_dir=None, metadata=None, vector_file=None,
               peepholes=False, bi=True, num_layers=2,
               attention=False, learning_rate=5e-5, args_config=None):
    model = ToxicityLSTM(csvs=csvs, batch_size=batch_size,
                         num_epochs=num_epochs, num_labels=num_labels,
                         comment_length=comment_length, peepholes=peepholes,
                         bi=bi, num_layers=num_layers, attention=attention,
                         learning_rate=learning_rate)
    train(model=model, verbose_freq=verbose_freq, save_freq=save_freq,
          meta=meta, log_dir=log_dir, model_dir=model_dir,
          metadata=metadata, word_vector_meta=word_vector_meta, word_vector_file=vector_file,
          vocab=vocab_size, args_config=args_config)


def train_word_vectors(csvs, vocab_size=18895, batch_size=2000,
                       num_epochs=160, embedding_size=100, verbose_freq=200,
                       save_freq=2000, meta=None, log_dir=None,
                       model_dir=None, metadata=None, nce_samples=64):
    model = WordEmbedding(csvs=csvs, vocab_size=vocab_size, batch_size=batch_size,
                          num_epochs=num_epochs, embedding_size=embedding_size,
                          nce_samples=nce_samples)
    train(model, verbose_freq=verbose_freq, save_freq=save_freq,
          meta=meta, log_dir=log_dir, model_dir=model_dir,
          metadata=metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', nargs='+',
                        help='Set training data csv files', required=True, dest='csvs')
    parser.add_argument('-b', '--batch', dest='batch_size',
                        help='Specify batch size', type=int, default=512)
    parser.add_argument('-e', '--epochs', dest='num_epochs',
                        help='Number of epochs', type=int, default=100)
    parser.add_argument('-v', '--vocab', dest='vocab_size',
                        help='Vocabulary size', type=int, required=True)
    parser.add_argument('--embedding', dest='embedding_size',
                        help='Embedding size', type=int, required=True)
    parser.add_argument('-l', '--labels', dest='num_labels',
                        help='Number of labels', type=int, default=6)
    parser.add_argument('-t', '--tlength', dest='comment_length',
                        help='Comment length', type=int, default=60)
    parser.add_argument('-s', '--sfreq', dest='save_freq',
                        help='Save frequency', type=int, default=1000)
    parser.add_argument('-m', '--mode', dest='mode',
                        help='Training mode', type=str, required=True)
    parser.add_argument('--metadata', dest='metadata', type=str,
                        help='Projector metadata file path')
    parser.add_argument('--samples', dest='nce_samples', default=64,
                        help='Negative sampling size', type=int)
    parser.add_argument('--word', dest='word_vector_meta', type=str,
                        help='Path to saved word vector variables')
    parser.add_argument('--meta', dest='meta', type=str,
                        help='Path to saved variables')
    parser.add_argument('--vector', dest='vector', type=str,
                        help='Path to pre trained word vectors.')
    parser.add_argument('--peepholes', dest='peepholes', action='store_true',
                        help='Use peephole connections in LSTM')
    parser.add_argument('--bi', dest='bi', action='store_true',
                        help='Use birnn')
    parser.add_argument('--stack', dest='num_layers', type=int,
                        help='Number of layers in lstm cell', default=2)
    parser.add_argument('--att', dest='attention', action='store_true',
                        help='Use attention model')
    parser.add_argument('--lr', dest='learning_rate', type=float,
                        help='Specify learning rate.', default=0.00005)
    parser.add_argument('--vfreq', dest='verbose_freq', type=int,
                        help='Specify verbose frequency.', default=100)

    args = parser.parse_args()
    args_dict = vars(args)

    if args.mode == 'cnn':
        train_cnn(csvs=args.csvs, batch_size=args.batch_size, num_epochs=args.num_epochs,
                  vocab_size=args.vocab_size, embedding_size=args.embedding_size, num_labels=args.num_labels,
                  comment_length=args.comment_length, save_freq=args.save_freq, metadata=args.metadata,
                  word_vector_meta=args.word_vector_meta, meta=args.meta, vector_file=args.vector,
                  args_config=args_dict)

    if args.mode == 'lstm':
        train_lstm(csvs=args.csvs, batch_size=args.batch_size, num_epochs=args.num_epochs,
                   vocab_size=args.vocab_size, num_labels=args.num_labels,
                   comment_length=args.comment_length, save_freq=args.save_freq, metadata=args.metadata,
                   word_vector_meta=args.word_vector_meta, meta=args.meta, vector_file=args.vector,
                   peepholes=args.peepholes, bi=args.bi, num_layers=args.num_layers,
                   attention=args.attention, learning_rate=args.learning_rate, verbose_freq=args.verbose_freq,
                   args_config=args_dict)

    if args.mode == 'emb':
        train_word_vectors(csvs=args.csvs, batch_size=args.batch_size,
                           num_epochs=args.num_epochs, vocab_size=args.vocab_size,
                           embedding_size=args.embedding_size, save_freq=args.save_freq,
                           metadata=args.metadata, nce_samples=args.nce_samples)

    if args.mode == 'test':
        test_cnn(csvs=args.csvs, meta=args.meta, batch_size=args.batch_size,
                 num_epochs=args.num_epochs, embedding_size=args.embedding_size,
                 vocab=args.vocab_size)

    if args.mode == 'testlstm':
        test_lstm(csvs=args.csvs, meta=args.meta, batch_size=args.batch_size,
                  num_epochs=args.num_epochs, embedding_size=args.embedding_size,
                  vocab=args.vocab_size, peepholes=args.peepholes, bi=args.bi,
                  num_layers=args.num_layers, attention=args.attention,
                  comment_length=args.comment_length, verbose_freq=args.verbose_freq)
