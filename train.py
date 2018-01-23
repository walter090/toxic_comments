import datetime
import os

import tensorflow as tf

from toxic_detection import ToxicityCNN


def train(csvs, batch_size, num_epochs,
          vocab_size, embedding_size, num_labels,
          verbose_freq=2000, save=True, restore=False,
          meta=None):
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

    if not restore:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
    else:
        restore_variables(meta, sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        while not coord.should_stop():
            sess.run(init_op)
            _, step, loss, summaries = sess.run([model.optimize, model.global_step,
                                                 model.loss, all_summaries])
            writer.add_summary(summaries, global_step=step)
            cur_time = datetime.datetime.now().isoformat('-')
            if step % verbose_freq == 0:
                print('{} - At step {}, loss {}'.format(cur_time, step, loss))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)

    if save:
        cur_step = tf.train.global_step(sess, model.global_step)
        saver.save(sess, save_path=model_dir, global_step=cur_step)

    sess.close()


def restore_variables(meta, sess):
    tf.reset_default_graph()
    imported = tf.train.import_meta_graph(meta)
    imported.restore(sess, tf.train.latest_checkpoint('./'))
