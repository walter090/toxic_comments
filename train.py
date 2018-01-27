import datetime
import os

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from toxic_detection import ToxicityCNN


def train(csvs, batch_size, num_epochs,
          vocab_size, embedding_size, num_labels,
          verbose_freq=2000, save_freq=10000, restore=False,
          meta=None, comment_length=60, log_dir=None,
          model_dir=None):

    model = ToxicityCNN(csvs=csvs, batch_size=batch_size,
                        num_epochs=num_epochs, vocab_size=vocab_size,
                        embedding_size=embedding_size, num_labels=num_labels,
                        comment_length=comment_length)

    timestamp = datetime.datetime.now().isoformat('_')
    save_dir = os.path.abspath(os.path.join(os.path.curdir, 'models_and_visual', timestamp))
    log_dir = os.path.join(save_dir, 'tensorboard') if not log_dir else log_dir
    model_dir = os.path.join(save_dir, 'saved_models') if not model_dir else model_dir

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
        embedding_projector.tensor_name = 'embeddings'
        projector.visualize_embeddings(writer, projector_config)

        sess.run(init_op)

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
                    saver = tf.train.Saver(var_list=tf.global_variables())
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
    imported = tf.train.import_meta_graph(meta)
    imported.restore(sess, tf.train.latest_checkpoint('./'))


if __name__ == '__main__':
    train(csvs=[os.path.abspath(os.path.join(os.path.curdir, 'dataset', 'processed',
                                             'translated', 'train.csv'))],
          batch_size=2000, num_epochs=100, vocab_size=18895,
          embedding_size=100, num_labels=6, comment_length=60)
