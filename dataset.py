import glob
import tensorflow as tf

BUFFER_SIZE = 800

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32) / 127.5 - 1
    w = tf.shape(image)[1] // 2
    label = image[:, :w, :]
    sketch = image[:, w:, :]
    return label, image


def get_dataset(dataset_path, batch_size, val_batch_size):
    train_path = glob.glob(dataset_path +'/train/*.png')
    train_dataset = tf.data.Dataset.from_tensor_slices(train_path)
    train_dataset = train_dataset.map(load_image, num_parallel_call=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    val_path = glob.glob(dataset_path + '/val/*.png')
    val_dataset = tf.data.Dataset.from_tensor_slices(val_path)
    val_dataset = val_dataset.map(load_image)
    val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(val_batch_size)

    return train_dataset, val_dataset