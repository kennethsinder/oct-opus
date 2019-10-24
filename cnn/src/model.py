import tensorflow as tf
from tensorflow.keras import datasets, layers, models


class CNN:
    def __init__(self):
        # build the model
        input = tf.keras.layers.Input(shape=[None, None, 1], name='input_image')

        conv_1 = layers.Conv2D(32, (3,3), activation='relu')(input)
        max_pool_1 = layers.MaxPool2D()(conv_1)

        conv_2 = layers.Conv2D(64, (3,3), activation='relu')(max_pool_1)
        max_pool_2 = layers.MaxPool2D()(conv_2)

        conv_3 = layers.Conv2D(128, (3,3), activation='relu')(max_pool_2)

        upsampling_1 = layers.UpSampling2D((2,2))(conv_3)
        concat_1 = layers.Concatenate()([conv_2, upsampling_1])
        conv_4 = layers.Conv2D(64, (3,3), activation='relu')(concat_1)

        upsampling_2 = layers.UpSampling2D((2,2))(conv_4)
        concat_2 = layers.Concatenate()([conv_1, upsampling_2])
        conv_5 = layers.Conv2D(64, (3,3), activation='relu')(concat_2)

        self.model = tf.keras.Model(inputs=input, outputs=conv_5)


    def train(self, train_dataset, num_epochs):
        self.model.compile(optimizer='Adam', loss='MeanAbsoluteError', metrics=['accuracy'])
        history = self.model.fit(
            x=train_dataset,
            batch_size=64,
            epochs=num_epochs,
            validation_split=0.2,
            verbose=2
        )
        return history


    def test(self, test_dataset):
        loss = model.evaluate(x=test_dataset, verbose=1)
        return loss


    def save_checkpoint(self, filepath):
        self.model.save(filepath)


    def load_checkpoint(self, filepath):
        self.model = model.load_model(filepath)
