import argparse
import os

import tensorflow as tf

import preprocess
from model.rnn import ToxicityLSTM


def save_model(save_dir, bi, num_layers,
               attention, peepholes, sentence_len,
               num_labels, vector_file, vocab_size,
               checkpoint_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    x_input = tf.placeholder(dtype=tf.int32, shape=(1, sentence_len),
                             name='x_input')
    word2id, embeddings = preprocess.build_vocab_from_file(vector_file,
                                                           limit=vocab_size,
                                                           save_dict=False)
    vocab_size, emb_size = len(embeddings), len(embeddings[0])

    model = ToxicityLSTM(comment_length=sentence_len, vocab_size=vocab_size,
                         embedding_size=emb_size, num_labels=num_labels,
                         batch_size=1, testing=True,
                         vec=embeddings, keep_prob=1,
                         bi=bi, num_layers=num_layers,
                         attention=attention, peepholes=peepholes)
    model.comment_batch = x_input
    prediction = model.prediction

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)

        checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess=sess, save_path=checkpoints)

        builder = tf.saved_model.builder.SavedModelBuilder(save_dir)

        legacy_init_op = tf.tables_initializer()
        input_info = tf.saved_model.utils.build_tensor_info(x_input)
        output_info = tf.saved_model.utils.build_tensor_info(prediction[-1])

        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            legacy_init_op=legacy_init_op,
            signature_def_map={
                'prediction': tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'sentence': input_info},
                    outputs={'output': output_info},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            }
        )

        builder.save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', dest='save_dir', required=True,
                        help='Directory to save the model.')
    parser.add_argument('--ckpt', dest='checkpoint_dir', required=True,
                        help='File name for meta file.')
    parser.add_argument('--bi', dest='bi', action='store_true',
                        help='Use flag to indicate that the model is bidirectional')
    parser.add_argument('--layers', dest='num_layers', required=True,
                        help='Number of stacked layers in the model.', type=int)
    parser.add_argument('--att', dest='attention', action='store_true',
                        help='Indicate the model uses attention.')
    parser.add_argument('--peep', dest='peepholes', action='store_true',
                        help='Indicate the model uses peepholes')
    parser.add_argument('--len', dest='sentence_len', required=True,
                        help='Length of each sequence')
    parser.add_argument('--label', dest='num_labels', required=True,
                        help='Number of labels')
    parser.add_argument('--vector', dest='vector_file', required=True,
                        help='Word vector file.')
    parser.add_argument('--vocab', dest='vocab_size', required=True,
                        help='Set vocabulary size', type=int)

    args = parser.parse_args()

    save_model(save_dir=args.save_dir, checkpoint_dir=args.checkpoint_dir, bi=args.bi,
               num_layers=args.num_layers, attention=args.attention, peepholes=args.peepholes,
               sentence_len=args.sentence_len, num_labels=args.num_labels, vector_file=args.vector_file,
               vocab_size=args.vocab_size)
