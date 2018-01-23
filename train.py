import datetime
import os

import tensorflow as tf

from toxic_detection import ToxicityCNN


def train(csvs, batch_size, num_epochs,
          vocab_size, embedding_size, num_labels,
          verbose_freq=2000, save=True):

    model = ToxicityCNN(csvs=csvs, batch_size=batch_size, num_epochs=num_epochs,
                        vocab_size=vocab_size, embedding_size=embedding_size, num_labels=num_labels)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(var_list=tf.all_variables())
    sess = tf.Session()
    timestamp = datetime.datetime.now().isoformat('_')

    all_summaries = tf.summary.merge_all()

    save_dir = os.path.abspath(os.path.join(os.path.curdir, 'models_and_visual', timestamp))
    log_dir = os.path.join(save_dir, 'tensorboard')
    model_dir = os.path.join(save_dir, 'saved_models')

    writer = tf.summary.FileWriter(log_dir, sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            sess.run(init_op)
            _, step, loss, summaries = sess.run([model.optimize, tf.train.global_step(sess, model.global_step),
                                                 model.loss, all_summaries])
            writer.add_summary(summaries, global_step=step)
            if step % verbose_freq == 0:
                print('At step {}, loss {}'.format(step, loss))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)

    # TODO save trained model
    if save:
        raise NotImplementedError

    sess.close()
