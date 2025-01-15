import tensorflow as tf
from tensorflow.keras import layers
from keras import Sequential
import numpy as np
from matplotlib import pyplot as plt


(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()


def image_preprocess(image):
    image = tf.expand_dims(image, axis=-1) 
    image = tf.image.resize(image, (28, 28)) 
    image = (image / 127.5) - 1 
    return image

train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
train_images = tf.map_fn(image_preprocess, train_images)
train = tf.data.Dataset.from_tensor_slices(train_images).batch(64).shuffle(1000).prefetch(tf.data.AUTOTUNE)

def generator(z):
    model = Sequential([
        layers.Dense(7 * 7 * 512, input_dim=z),
        layers.Reshape((7, 7, 512)),
        layers.Conv2DTranspose(512, kernel_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(256, kernel_size=3, strides=2, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh')
    ])
    return model

def discriminator(img_shape):
    model = Sequential([
        layers.Conv2D(256, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.Conv2D(32, kernel_size=3, strides=2, padding='same'),
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def gen_images(generator, z, num):
    noise = tf.random.normal([num, z])
    generated_images = generator.predict(noise, verbose=0)
    generated_images = (generated_images + 1) / 2.0 

    fig, axs = plt.subplots(1, num, figsize=(15, 15))
    for i in range(num):
        axs[i].imshow(generated_images[i].squeeze(), cmap='gray')
        axs[i].axis('off')
    plt.show()


def gan_train(epochs, batch_size, z):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for real_images in train:
            batch_size = real_images.shape[0]

            noise = tf.random.normal([batch_size, z])
            fake_images = gen(noise, training=False)
            
            # Prepare discriminator data
            x = tf.concat([real_images, fake_images], axis=0)
            y = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
            
            # Train discriminator
            d_metrics = dis.train_on_batch(x, y)
            d_loss, d_acc = d_metrics[0], d_metrics[1] 
            
            noise = tf.random.normal([batch_size, z])
            y_gen = tf.ones((batch_size, 1)) 
            g_loss = gan.train_on_batch(noise, y_gen)
        
        print(f"Epoch {epoch + 1}: Discriminator Loss: {d_loss}, Accuracy: {d_acc}, Generator Loss: {g_loss}")
        
        if (epoch + 1) % 2 == 0:
            gen_images(gen, z, num=4)


z = 100
image_shape = (28, 28, 1) 

gen = generator(z)
dis = discriminator(image_shape)

dis.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

dis.trainable = False
gan_input = tf.keras.Input(shape=(z,))
gan_output = dis(gen(gan_input))
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

gan_train(epochs=10, batch_size=32, z=z)
