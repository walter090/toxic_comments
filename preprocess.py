import tensorflow as tf


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
