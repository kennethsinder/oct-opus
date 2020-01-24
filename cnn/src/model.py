import tensorflow as tf
from tensorflow.keras import layers, models, losses

from params import IMAGE_DIM


class CNN:
    def __init__(self):
        input = tf.keras.layers.Input(shape=(IMAGE_DIM, IMAGE_DIM, 1), name='input')

        conv_1 = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv_1')(input)
        max_pool_1 = layers.MaxPool2D(name='max_pool_1')(conv_1)

        conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv_2')(max_pool_1)
        max_pool_2 = layers.MaxPool2D(name='max_pool_2')(conv_2)

        conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same', name='conv_3')(max_pool_2)

        upsampling_1 = layers.UpSampling2D((2,2), name='upsampling_1')(conv_3)
        upconv_1 = layers.Conv2D(64, (2,2), padding='same', name='upconv_1')(upsampling_1)

        concat_1 = layers.Concatenate(name='concat_1')([conv_2, upconv_1])
        conv_4 = layers.Conv2D(64, (3,3), activation='relu', padding='same', name='conv_4')(concat_1)

        upsampling_2 = layers.UpSampling2D((2,2), name='upsampling_2')(conv_4)
        upconv_2 = layers.Conv2D(32, (2,2), padding='same', name='upconv_2')(upsampling_2)

        concat_2 = layers.Concatenate(name='concat_2')([conv_1, upconv_2])
        conv_5 = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv_5')(concat_2)

        output = layers.Conv2D(1, (1,1), activation='linear', padding='same', name='output')(conv_5)
        self.model = tf.keras.Model(inputs=input, outputs=output)


    def train(self, train_dataset, num_epochs):
        self.model.compile(
            optimizer='Adam',
            loss=losses.MeanAbsoluteError(),
            metrics=['accuracy'],
        )
        history = self.model.fit(
            x=train_dataset,
            epochs=num_epochs,
            verbose=2,
        )
        return history


    def test(self, test_dataset):
        loss = model.evaluate(x=test_dataset, verbose=1)
        return loss


    def summary(self):
        self.model.summary()


    # def save_checkpoint(self, filepath):
    #     self.model.save(filepath)
    #
    #
    # def load_checkpoint(self, filepath):
    #     self.model = model.load_model(filepath)
