import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import numpy as np
import time

class Linear_QNet():
    def __init__(self, input_shape, hidden_shape, output_shape):
        self.model = self.build_model(input_shape, hidden_shape, output_shape)
        self.lr = 0.01
        self.gamma = 0.9

        self.model.compile(Adam(learning_rate=self.lr), loss="mean_squared_error", metrics=['mean_squared_error'])


    def build_model(self, input_shape, hidden_shape, output_shape):
        input = keras.Input(shape=(input_shape,), name="digits")
        hidden = layers.Dense(hidden_shape, activation="relu")(input)
        output = layers.Dense(output_shape, name="predictions")(hidden)
        model = keras.Model(inputs=input, outputs=output)
        model.summary()
        return model

    def predict(self, input):
        return self.model.predict(input)

    def train_step(self, state, action, reward, next_state, done):
        state = tf.constant(state, dtype=tf.float32)
        next_state = tf.constant(next_state, dtype=tf.float32)
        action = tf.constant(action, dtype=tf.int64)
        reward = tf.constant(reward, dtype=tf.float64)

        if len(state.shape) == 1:
            # (1, x)
            state = tf.expand_dims(state, 0)
            next_state = tf.expand_dims(next_state, 0)
            action = tf.expand_dims(action, 0)
            reward = tf.expand_dims(reward, 0)
            done = (done, )


        # 1: predicted Q values with current state
        pred = self.model(state, training=True).numpy()

        target = pred.copy()

        Q_new = reward

        if len(done) == 1:
            if not done[0]:
                next_pred = self.model.predict(next_state)
                Q_new = reward + self.gamma * next_pred.max()

            mut_index = tf.math.argmax(action[0])
            target[0][mut_index] = Q_new
        else:
            start_time = time.time()
            for idx in range(len(done)):
                Q_new = reward[idx]
                if not done[idx]:

                    n_state = tf.expand_dims(next_state[idx], 0)
                    next_pred = self.model(n_state, training=True).numpy()  # Runs VERY slowly
                    Q_new = reward[idx] + self.gamma * next_pred[0].max()


                action_value = action[idx]
                mut_index = tf.math.argmax(action_value)
                target[idx][mut_index] = Q_new
            print("--- %s seconds ---" % (time.time() - start_time))


        self.model.train_on_batch(state, target)





