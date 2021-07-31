from tensorflow.keras.losses import BinaryCrossEntropy, MeanAbsoluteError
import tensorflow as tf

cross_entropy = BinaryCrossentropy(from_logits=True)
l1_loss = MeanAbsoluteError()

def discriminator_loss(y_true, y_generated):
    real_loss = cross_entropy(tf.ones_like(y_true), y_true)
    fake_loss = cross_entropy(tf.zeros_like(y_generated), y_generated)
    total_loss = real_loss + fake_loss
    return tf.math.divide(total_loss, tf.constant(2.0, shape=total_loss.shape))


def generator_loss(y_true, y_generated, disc_gen_out):
    gan = cross_entropy(tf.ones_like(disc_gen_out), disc_gen_out)
    l1 = l1_loss(y_true, y_generated)
    total = l1 * LAMBDA + gan
    return total, l1, gan