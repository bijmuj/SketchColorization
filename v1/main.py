from dataset import get_dataset
from model import get_vgg, vgg_preprocess, discriminator_model, generator_model
from utils import display, log
from losses import generator_loss, discriminator_loss
import tensorflow as tf
import json
import time
import wandb


@tf.function
def train_step(labels, sketches):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        vgg_output = vgg_model(vgg_preprocess(labels), training=False)
        generated_images = generator([sketches, vgg_output], training=True)
        
        real_outputs = discriminator(labels, training=True)
        fake_outputs = discriminator(generated_images, training=True)

        gen_loss_total, gen_loss_l1, gen_loss_gan = generator_loss(labels, generated_images, fake_outputs)
        disc_loss = discriminator_loss(real_outputs, fake_outputs)

    gradients_of_generator = gen_tape.gradient(gen_loss_total, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss_total, disc_loss


def fit(train_dataset, val_dataset):
    discriminator = discriminator_model()
    generator = generator_model()
    generator_optimizer = tf.keras.optimizers.Adam(config['lr'], beta_1=0.5, beta_2=0.999)
    discriminator_optimizer = tf.keras.optimizers.Adam(config['lr'], beta_1=0.5, beta_2=0.999)

    checkpoint = tf.train.Checkpoint(discriminator=discriminator,
                                    generator=generator,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator_optimizer=generator_optimizer)
    manager = tf.train.CheckpointManager(checkpoint, config['ckpt'], max_to_keep=3)

    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        start = int(manager.latest_checkpoint.split('-')[-1])
    else:
        print("Initializing from scratch.")
        start = 0

    vgg_model = get_vgg()

    val_iter = iter(val_dataset)
    for epoch in range(start, EPOCHS):
        g_losses = []
        d_losses = []
        print(f'Epoch: {epoch + 1} of {EPOCHS}')
        named_tuple = time.localtime()
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        print("Started: " + time_string)

        for n, (labels, sketches) in train_dataset.enumerate():
            g_loss, d_loss = train_step(labels, sketches)
            g_losses.append(g_loss)
            d_losses.append(d_loss)
        g_loss = np.mean(np.array(g_losses))
        d_loss = np.mean(np.array(d_losses)) * 2

        named_tuple = time.localtime()
        time_string = time.strftime("%m/%d/%Y, %H:%M:%S", named_tuple)
        print("Ended: " + time_string)

        print(f'Generator loss: {g_loss}\nDiscriminator loss: {d_loss}')
        path = manager.save()
        print(f'Checkpoint saved at {checkpoint_dir}')

        labels, sketches = val_iter.next()
        vgg_out = vgg_model.predict(vgg_preprocess(labels))
        imgs = generator([sketches, vgg_out], training=False)
        log(g_loss, d_loss, labels, sketches, imgs)


def main():
    with open('config.json', 'r') as f:
        config = json.load(f)

    if "wandb" in config:
        wandb.login()
        run = wandb.init(project="sketch_colorization", config=config)

    train_dataset, val_dataset = get_dataset(config["dataset"], config["batch_size"], config["val_batch_size"])

    fit(train_dataset, val_dataset, start)
