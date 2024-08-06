import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tqdm.notebook import tqdm


# Define Generator 
def Generator(img_shape, dim_latent, g_dims=[128,256,512,1024]):
    # Define block component
    def block(x, out_feat, normalize=True):
        x = layers.Dense(units=out_feat)(x)
        if normalize:
            x = tf.keras.layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        return x

    input_G = layers.Input(dim_latent)
    x = block(input_G, g_dims[0], normalize=False)
    for i in range(len(g_dims)-1):
        x = block(x, g_dims[i+1])

    x = layers.Dense(units=int(np.prod(img_shape)), activation=tf.nn.tanh)(x)
    imgs = layers.Reshape([*img_shape])(x)
    model = tf.keras.Model(inputs=input_G, outputs=imgs)

    return model


# Define Discriminator
def Discriminator(img_shape, d_dims=[512, 256]):
    def block(x, out_feat):
        x = layers.Dense(units=out_feat)(x)
        x = layers.LeakyReLU()(x)
        return x

    input_D = layers.Input(img_shape)
    x = layers.Flatten()(input_D)
    x = block(x, d_dims[0])
    for i in range(len(d_dims)-1):
        x = block(x, d_dims[i+1])
    pred = layers.Dense(units=1, activation=tf.nn.sigmoid)(x)
    model = tf.keras.Model(inputs=input_D, outputs=pred)

    return model


def Train(epoch, dim_latent, dataloader, G, D, optimizer_G, optimizer_D, verbose_freq=10):
    for j in tqdm(range(epoch)):
        for _, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.shape[0]

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                # Generate noise vector & get generated images from generator
                z = tf.random.normal([batch_size, dim_latent])
                gen_imgs = G(z, training=True)

                real_pred = D(real_imgs, training=True)
                fake_pred = D(gen_imgs, training=True)

                # Compute loss for generator & discriminator respectively
                # Loss_G = E[log(D(G(z))]
                loss_G = -tf.math.reduce_mean(tf.math.log(fake_pred),0)
                # Loss_D = E[log(D(x))]+E[log(1-D(G(z)))] 
                loss_D = -tf.math.reduce_mean(tf.math.log(real_pred)+tf.math.log(1-fake_pred),0)

            grad_D = disc_tape.gradient(loss_D, D.trainable_variables)
            grad_G = gen_tape.gradient(loss_G, G.trainable_variables)

            optimizer_D.apply_gradients(zip(grad_D, D.trainable_variables))
            optimizer_G.apply_gradients(zip(grad_G, G.trainable_variables))

        if (j+1) % verbose_freq == 0:
            print(f"Epoch {j+1} / D loss: {loss_D.numpy()[0]:.3f} / G loss: {loss_G.numpy()[0]:.3f}")