# model architecture will be defined here

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    LSTM, 
    Activation, 
    Dropout, 
    Dense, 
    RepeatVector, 
    Reshape, 
    Embedding,
    Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalCrossentropy as cce_metric, CategoricalAccuracy

import numpy as np

class GenPhiloText(tf.keras.Model):
    def __init__(self, emb_dim=32, n_a=128, n_unique=26, T_x=50, keep_prob=1, lambda_=1):
        super(GenPhiloText, self).__init__()

        # instantiate layers
        self.lstm_cell = LSTM(units=n_a, return_state=True)
        self.dense_layer = Dense(units=n_unique)
        self.out_layer = Activation(activation=tf.nn.softmax)
        self.drop_layer = Dropout(1 - keep_prob)

        self.char_emb_layer = Embedding(n_unique, emb_dim, embeddings_regularizer=L2(lambda_))

        # utility layers
        self.reshape_layer = Reshape(target_shape=(1, n_unique))

        # number of time steps or length of longest sequences/training example
        self.n_time_steps = T_x

    def call(self, inputs, **kwargs):
        # get batch of training examples, hidden state, and cell 
        # state inputs by destructuring inputs
        X, h_0, c_0 = inputs[0], inputs[1], inputs[2]
        h = h_0
        c = c_0

        # print(X[:, 1, :])
        # this will keep track of each predicted y output of each LSTM cell
        outputs = []

        # define architecture
        for t in range(self.n_time_steps):
            # get slice of input such that shape goes from
            # m, T_x, n_unique to m, n_unique, since we are taking
            # a single matrix from a single time step
            x = X[:, t, :]

            # because each timestep takes in a (m, 1, n_unique)
            # input we must reshape our input x at timestep t
            x = self.reshape_layer(x)

            # pass the input x to the LSTM cell as well as the 
            # hidden and cell states that will constantly change
            whole_seq_y, h, c = self.lstm_cell(inputs=x, initial_state=[h, c])

            # pass final hidden state to dropout layer
            # during training
            if kwargs['training'] == True:
                drop = self.drop_layer(h)

            # pass the hidden state to the dense layer
            x = self.dense_layer(drop if kwargs['training'] == True else h)
            out = self.out_layer(x)
            print(out)

            # when all outputs are collected this will 
            # have dimensionality (T_y, m, n_unique)
            outputs.append(out)

        return outputs
    
def load_alt_model(emb_dim=32, n_a=128, n_unique=26, T_x=50, keep_prob=1, lambda_=1):
    """
    args:
        emb_dim -
        n_a - 
        n_unique - 
        T_x -
        keep_prob
        lambda_
    """
    X = Input(shape=(T_x, n_unique))
    h_0 = Input(shape=(n_a,), name='init_hidden_state')
    c_0 = Input(shape=(n_a,), name='init_cell_state')

    h = h_0
    c = c_0

    outputs = []

    # define architecture
    for t in range(T_x):
        x = X[:, t, :]

        x = Reshape(target_shape=(1, n_unique))(x)
        whole_seq_y, h, c = LSTM(units=n_a, return_state=True)(inputs=x, initial_state=[h, c])
        x = Dropout(1 - keep_prob)(h)
        x = Dense(units=n_unique)(x)
        out = Activation(activation=tf.nn.softmax)(x)

        outputs.append(out)

    return Model(inputs=[X, h_0, c_0], outputs=outputs)

def load_inf_model():
    pass



if __name__ == "__main__":
    # (m, T_x, n_features)
    m = 100
    T_x = 50
    n_unique = 26
    n_a = 128
    X = np.random.rand(m, T_x, n_unique)

    # we have to match the output of the prediction of our 
    # model which is a list of (100, 26) values. So instead of a 3D matrixc
    # we create a list fo 2D matrices of shape (100, 26)
    Y = [np.random.rand(m, n_unique) for _ in range(T_x)]

    # one hot encode our dummy (T_y, m, n_unique) probabilities
    Y = [tf.one_hot(tf.argmax(y, axis=1), depth=n_unique) for y in Y]
    print(Y)

    # initialize hidden and cell states to shape (m, n_units)
    h_0 = np.zeros(shape=(m, n_a))
    c_0 = np.zeros(shape=(m, n_a))

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    loss = cce_loss()
    metrics = [CategoricalAccuracy(), cce_metric()]

    # instantiate custom model
    # model = GenPhiloText(n_a=n_a, n_unique=n_unique, T_x=T_x)
    model = load_alt_model(n_a=n_a, n_unique=n_unique, T_x=T_x, keep_prob=0.7)

    # compile 
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model.summary()

    # train
    model.fit([X, h_0, c_0], Y, epochs=100)
    