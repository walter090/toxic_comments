import argparse
import datetime
import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from model.toxic_detection import ToxicityCNN
from model.skip_gram import WordEmbedding


def train(model, verbose_freq=200, save_freq=2000,
          restore=False, meta=None, log_dir=None,
          model_dir=None, metadata=None):
    timestamp = datetime.datetime.now().isoformat('_')
    save_dir = os.path.abspath(os.path.join(os.path.curdir, 'models_and_visual', timestamp))
    log_dir = os.path.join(save_dir, 'tensorboard') if not log_dir else log_dir
    model_dir = os.path.join(save_dir, 'saved_models') if not model_dir else model_dir

    model_embeddings = model.embeddings  # 18895
    model_grads, model_optimization = model.optimize
    model_step = model.global_step
    model_loss = model.loss

    tf.summary.scalar('loss', model_loss)

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

        if restore:
            restore_variables(meta, sess)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                _, step, loss, summaries = sess.run([model_optimization, model_step,
                                                     model_loss, all_summaries])

                writer.add_summary(summaries, global_step=step)

                cur_time = datetime.datetime.now().isoformat('-')
                if step % verbose_freq == 0:
                    print('{} - At step {}, loss {}'.format(cur_time, step, loss))
                if step % save_freq == 0:
                    saver.save(sess, save_path=model_dir, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()

        coord.join(threads)
        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.save(sess, save_path=model_dir)


def restore_variables(meta, sess):
    tf.reset_default_graph()
    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path=meta)


def restore_word_vectors(meta, sess):
    tf.reset_default_graph()
    saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                      'embedding'))
    saver.restore(sess=sess, save_path=meta)


def train_cnn(csvs, vocab_size=18894, batch_size=2000,
              num_epochs=160, embedding_size=100, num_labels=6,
              comment_length=60, verbose_freq=200, save_freq=2000,
              restore=False, meta=None, log_dir=None,
              model_dir=None, metadata=None):
    model = ToxicityCNN(csvs=csvs, batch_size=batch_size,
                        num_epochs=num_epochs, vocab_size=vocab_size,
                        embedding_size=embedding_size, num_labels=num_labels,
                        comment_length=comment_length)
    train(model=model, verbose_freq=verbose_freq, save_freq=save_freq,
          restore=restore, meta=meta, log_dir=log_dir,
          model_dir=model_dir, metadata=metadata)


def train_word_vectors(csvs, vocab_size=18894, batch_size=2000,
                       num_epochs=160, embedding_size=100, verbose_freq=200,
                       save_freq=2000, restore=False, meta=None,
                       log_dir=None, model_dir=None, metadata=None):
    model = WordEmbedding(csvs=csvs, vocab_size=vocab_size, batch_size=batch_size,
                          num_epochs=num_epochs, embedding_size=embedding_size)
    train(model, verbose_freq=verbose_freq, save_freq=save_freq,
          restore=restore, meta=meta, log_dir=log_dir,
          model_dir=model_dir, metadata=metadata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data', nargs='+',
                        help='Set training data csv files', required=True, dest='csvs')
    parser.add_argument('-b', '--batch', dest='batch_size',
                        help='Specify batch size', type=int, default=2000)
    parser.add_argument('-e', '--epochs', dest='num_epochs',
                        help='Number of epochs', type=int, default=160)
    parser.add_argument('-v', '--vocab', dest='vocab_size',
                        help='Vocabulary size', type=int, required=True)
    parser.add_argument('--embedding', dest='embedding_size',
                        help='Embedding size', type=int, default=100)
    parser.add_argument('-l', '--labels', dest='num_labels',
                        help='Number of labels', type=int, default=6)
    parser.add_argument('-t', '--tlength', dest='comment_length',
                        help='Comment length', type=int, default=60)
    parser.add_argument('-s', '--sfreq', dest='save_freq',
                        help='Save frequency', type=int, default=1000)
    parser.add_argument()

    args = parser.parse_args()

    train_cnn(csvs=args.csvs, batch_size=args.batch_size, num_epochs=args.num_epochs,
              vocab_size=args.vocab_size, embedding_size=args.embedding_size, num_labels=args.num_labels,
              comment_length=args.comment_length, save_freq=args.save_freq)
