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
    Input,
    BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalCrossentropy as cce_metric, CategoricalAccuracy

import numpy as np



@tf.keras.utils.register_keras_serializable()
class GenPhiloText(tf.keras.Model):
    def __init__(self, emb_dim=32, n_a=128, n_unique=26, T_x=50, **kwargs):
        super(GenPhiloText, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_a = n_a
        self.n_unique = n_unique

        # number of time steps or length of longest sequences/training example
        self.n_time_steps = T_x

        # instantiate layers
        self.char_emb_layer = Embedding(n_unique, emb_dim, name='char-emb-layer')
        self.lstm_cell = LSTM(units=n_a, return_state=True, name='lstm-cell')
        self.dense_layer = Dense(units=n_unique, name='dense-layer')
        self.out_layer = Activation(activation=tf.nn.softmax)
        
        # utility layers
        self.reshape_layer = Reshape(target_shape=(1, emb_dim), name='reshape-layer')
        self.norm_layer = BatchNormalization(name='norm-layer')

    def call(self, inputs):
        # get batch of training examples, hidden state, and cell 
        # state inputs by destructuring inputs
        X, h_0, c_0 = inputs
        h = h_0
        c = c_0

        # print(X[:, 1, :])
        # this will keep track of each not predicted y outputs
        # but the preliminary logits outputted by the dense or batch norm
        # layer before it goes to the activation layer
        out_logits = []

        # define architecture

        # convert (m, T_x) inputs to embeddings of (m, T_x, n_feawtures)
        embeddings = self.char_emb_layer(X)

        # loop over each timestep
        for t in range(self.n_time_steps):
            

            # from here get slice of the embeddings such that shape 
            # goes from (m, T_x, n_features) to (m, n_features), 
            # since we are taking a single matrix from a single time step
            x_t = embeddings[:, t, :]

            # because each timestep takes in a (m, 1, n_unique)
            # input we must reshape our input x at timestep t
            x_t = self.reshape_layer(x_t)

            # pass the input x to the LSTM cell as well as the 
            # hidden and cell states that will constantly change
            states = self.lstm_cell(inputs=x_t, initial_state=[h, c])
            _, h, c = states


            # pass final hidden state to the dense layer
            # pass the hidden state to the dense layer
            z_t = self.dense_layer(h)
            z_t = self.norm_layer(z_t)

            # when all outputs are collected this will 
            # have dimensionality (T_y, m, n_unique)
            out_logits.append(z_t)

        return out_logits
    
    def get_config(self):
        config = super(GenPhiloText, self).get_config()
        config['emb_dim'] = self.emb_dim
        config['n_a'] = self.n_a
        config['n_unique'] = self.n_unique
        config['T_x'] = self.n_time_steps

        return config

def load_alt_model_a(n_unique, T_x, emb_dim=32, n_a=128):
    """
    args:
        emb_dim -
        n_a - 
        n_unique - 
        T_x -
    """

    # instantiate sequential model
    model = Sequential()

    # (m, T_x)
    model.add(Input(shape=(T_x, )))

    # (m, T_x, n_unique)
    model.add(Embedding(n_unique, emb_dim))

    # (m, T_x, n_a)
    model.add(LSTM(units=n_a, return_sequences=True))

    # (m, n_a)
    model.add(LSTM(units=n_a, return_sequences=False))

    # (m, n_unique)
    model.add(Dense(units=n_unique))
    model.add(BatchNormalization())
    model.add(Activation(activation=tf.nn.softmax))

    return model

def load_alt_model_b(n_unique, T_x, emb_dim=32, n_a=128):
    """
    args:
        emb_dim -
        n_a - 
        n_unique - 
        T_x -
    """

    # define shape of batch of inputs including 
    # hidden and cell states
    X = Input(shape=(T_x,), name='X')
    h_0 = Input(shape=(n_a,), name='init-hidden-state')
    c_0 = Input(shape=(n_a,), name='init-cell-state')

    # pass input X inside embedding layer such that X which is (m, T_x) 
    # is transformed to (m, T_x, n_features) which the LSTM layer can accept
    embeddings = Embedding(n_unique, emb_dim, name='character-lookup')(X)

    # define reshaper, Lstm, dense, norm, and act layers here
    # since each layer will only be used once this is because we need
    # the weights of the layers at each time step to be the same and not
    # constantly changing
    reshape_layer = Reshape(target_shape=(1, emb_dim), name='reshape-layer')
    lstm_cell = LSTM(units=n_a, return_state=True, name='lstm-cell')
    dense_layer = Dense(units=n_unique, name='dense-layer')
    norm_layer = BatchNormalization(name='norm-layer')

    # initialize hidden and cell states
    h = h_0
    c = c_0

    # this will keep track of each not predicted y outputs
    # but the preliminary logits outputted by the dense or batch norm
    # layer before it goes to the activation layer
    out_logits = []

    # define architecture
    for t in range(T_x):
        # from here get slice of the embeddings such that shape 
        # goes from (m, T_x, n_features) to (m, n_features), 
        # since we are taking a single matrix from a single time step
        x_t = embeddings[:, t, :]

        # because each timestep takes in a (m, 1, n_features)
        # input we must reshape our input x at timestep
        # from (m, n_features) to (m, 1, n_features)
        x_t = reshape_layer(x_t)

        # pass the input x to the LSTM cell as well as the 
        # hidden and cell states that will constantly change
        states = lstm_cell(inputs=x_t, initial_state=[h, c])
        _, h, c = states

        # pass the hidden state to the dense 
        # layer and then normalize after
        z_t = dense_layer(h)
        z_t = norm_layer(z_t)

        out_logits.append(z_t)

    return Model(inputs=[X, h_0, c_0], outputs=out_logits)

def load_inf_model(char_emb_layer, lstm_cell, dense_layer, norm_layer, char_to_idx, T_x: int=100, chars_to_skip: list=['[UNK]']):
    """
    args:
        char_emb_layer - 
        lstm_cell - 
        dense_layer - 
        norm_layer - 
    """
    # retrieve number of unique chars from dense layer
    n_chars = dense_layer.units
    n_a = lstm_cell.units
    print(n_chars)
    print(n_a)

    # ids to skip has shape (1, 1)
    ids_to_skip = char_to_idx(chars_to_skip)[:, None]
    sparse_mask_vector = tf.SparseTensor(values=[float('-inf')] * len(ids_to_skip), indices=ids_to_skip, dense_shape=[n_chars])
    dense_mask_vector = tf.reshape(tf.sparse.to_dense(sparse_mask_vector), shape=(1, -1))
    print(dense_mask_vector)

    # define shape of batch of inputs including hidden and cell 
    # states. Note in the prediction stage X will only be a (1, 1)
    # input representing one example and 1 timestep
    X = Input(shape=(1,))
    h_0 = Input(shape=(n_a,), name='init_hidden_state')
    c_0 = Input(shape=(n_a,), name='init_cell_state')

    idx = -1

    # a flag that represents how many newlines we have left before
    # sequence generation stops
    num_newlines = 2

    # extract learned embeddings from embedding matrix.
    # once (1, 1) input is fed output embeddings will now be 
    # (1, 1, 32) 32 can be variable depending on the initially 
    # set number of features during training
    x_t = char_emb_layer(X)
    h = h_0
    c = c_0

    i = 0
    while num_newlines != 0 and i != T_x:
        # since embedding x_t is already a (1, 1, 32) input
        # we can feed it directly to our lstm_cell
        states = lstm_cell(inputs=x_t, initial_state=[h, c])
        _, h, c = states

        # pass the hidden state to the dense 
        # layer and then normalize after
        z_t = dense_layer(h)
        z_t = norm_layer(z_t)

        # because tensor after norm layer is (1, 57) for example should
        # our n unique chars be 57, we must also have our mask tensor to
        # be of the same shape
        


if __name__ == "__main__":
    # (m, T_x, n_features)
    m = 100
    T_x = 50
    n_unique = 26
    n_a = 128
    emb_dim = 32
    X = np.random.randint(0, n_unique, size=(m, T_x))

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
    loss = cce_loss(from_logits=True)
    metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]

    # instantiate custom model
    model = GenPhiloText(n_a=n_a, n_unique=n_unique, T_x=T_x)
    # model = load_alt_model_a(n_a=n_a, n_unique=n_unique, T_x=T_x)
    # model = load_alt_model_b(n_a=n_a, n_unique=n_unique, T_x=T_x)

    # compile 
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    # model.summary()

    # train
    model.fit([X, h_0, c_0], Y, epochs=100, verbose=2)
    
    # save model
    model.save_weights('../saved/weights/test_model_gen_philo_text.h5', save_format='h5')
    # model.save('../saved/models/test_model_a.h5', save_format='h5')
    # model.save('../saved/models/test_model_b.h5', save_format='h5')
    