import tensorflow as tf


def file_read_op(file_names):
    """Read csv files

    Args:
        file_names: Python list of file names

    Returns:

    """
    reader = tf.TextLineReader(skip_header_lines=1)
    queue = tf.train.string_input_producer(file_names)

    _, value = reader.read(queue)
    record_defaults = [[''], [0], [0], [0], [0], [0], [0]]
    cols = tf.decode_csv(value, record_defaults=record_defaults)

    comment_text = cols[0]
    toxcity = tf.stack(cols[1:])
    return comment_text, toxcity
