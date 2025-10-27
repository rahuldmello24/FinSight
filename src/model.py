import tensorflow as tf

def build_lstm_model(time_steps, num_features, hidden_size, num_layers, dropout):
    """ Builds lstm model using Tensorflow"""

    model = tf.keras.Sequential()

    # input layer
    model.add(tf.keras.layers.Input(shape=(time_steps, num_features)))

    for i in range(num_layers):
        return_sequences = i < num_layers - 1 # only return sequences for all but last layer
        model.add(tf.keras.layers.LSTM(hidden_size, return_sequences=return_sequences,
                                       dropout=dropout))

        # final output layer
        model.add(tf.keras.layers.Dense(1, activation='linear'))

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse')

        return model



