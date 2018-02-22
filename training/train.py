import argparse
import datetime
import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import preprocess
from model.toxic_detection import ToxicityCNN
from model.skip_gram import WordEmbedding


def train(model, verbose_freq=200, save_freq=2000,
          meta=None, log_dir=None, model_dir=None,
          metadata=None, word_vector_meta=None, word_vector_file=None):
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

    Returns:
        None
    """
    timestamp = datetime.datetime.now().isoformat('_')
    save_dir = os.path.abspath(os.path.join(os.path.curdir, 'models_and_visual', timestamp))
    log_dir = os.path.join(save_dir, 'tensorboard') if not log_dir else log_dir
    model_dir = os.path.join(save_dir, 'saved_models') if not model_dir else model_dir

    if word_vector_file:
        word2id, vec = preprocess.build_vocab_from_file(word_vector_file)
        model.provide_vector(vec)

    model_grads, model_optimization = model.optimize
    model_step = model.global_step
    model_loss = model.loss
    model_auc = model.metric
    model_embeddings = model.embeddings[0]  # 18895

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
    model_loss = model.loss
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
                loss, auc = sess.run([model_loss, model_auc])
                if step % verbose_freq == 0:
                    print('At step {}: loss {}, AUC {}'.format(step, loss, auc))
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


def test_cnn(csvs, meta, batch_size=516,
             num_epochs=1, embedding_size=300):
    model = ToxicityCNN(csvs=csvs, vocab_size=18895, batch_size=batch_size,
                        num_epochs=num_epochs, embedding_size=embedding_size, num_labels=6,
                        comment_length=60, testing=True)
    test(model, meta=meta)


def train_cnn(csvs, vocab_size=18895, batch_size=512,
              num_epochs=150, embedding_size=100, num_labels=6,
              comment_length=60, verbose_freq=200, save_freq=2000,
              word_vector_meta=None, meta=None, log_dir=None,
              model_dir=None, metadata=None, vector_file=None):
    model = ToxicityCNN(csvs=csvs, batch_size=batch_size,
                        num_epochs=num_epochs, vocab_size=vocab_size,
                        embedding_size=embedding_size, num_labels=num_labels,
                        comment_length=comment_length)
    train(model=model, verbose_freq=verbose_freq, save_freq=save_freq,
          meta=meta, log_dir=log_dir, model_dir=model_dir,
          metadata=metadata, word_vector_meta=word_vector_meta, word_vector_file=vector_file)


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
                        help='Vocabulary size', type=int, required=False)
    parser.add_argument('--embedding', dest='embedding_size',
                        help='Embedding size', type=int, required=True)
    parser.add_argument('-l', '--labels', dest='num_labels',
                        help='Number of labels', type=int, default=6)
    parser.add_argument('-t', '--tlength', dest='comment_length',
                        help='Comment length', type=int, default=60)
    parser.add_argument('-s', '--sfreq', dest='save_freq',
                        help='Save frequency', type=int, default=1000)
    parser.add_argument('-m', '--mode', dest='mode',
                        help='Training mode', type=str, default='emb')
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

    args = parser.parse_args()

    if args.mode == 'cnn':
        train_cnn(csvs=args.csvs, batch_size=args.batch_size, num_epochs=args.num_epochs,
                  vocab_size=args.vocab_size, embedding_size=args.embedding_size, num_labels=args.num_labels,
                  comment_length=args.comment_length, save_freq=args.save_freq, metadata=args.metadata,
                  word_vector_meta=args.word_vector_meta, meta=args.meta, vector_file=args.vector)

    if args.mode == 'emb':
        train_word_vectors(csvs=args.csvs, batch_size=args.batch_size,
                           num_epochs=args.num_epochs, vocab_size=args.vocab_size,
                           embedding_size=args.embedding_size, save_freq=args.save_freq,
                           metadata=args.metadata, nce_samples=args.nce_samples)

    if args.mode == 'test':
        test_cnn(csvs=args.csvs, meta=args.meta, batch_size=args.batch_size,
                 num_epochs=args.num_epochs, embedding_size=args.embedding_size)
