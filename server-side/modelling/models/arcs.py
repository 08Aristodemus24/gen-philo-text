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
    BatchNormalization,
    Add)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.metrics import CategoricalCrossentropy as cce_metric, CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np



@tf.keras.utils.register_keras_serializable()
class GenPhiloTextA(tf.keras.Model):
    def __init__(self, emb_dim=32, n_a=128, n_unique=26, dense_layers_dims=[26], lambda_=1, drop_prob=0.0, normalize=False, **kwargs):
        super(GenPhiloTextA, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_a = n_a
        self.n_unique = n_unique
        self.dense_layers_dims = dense_layers_dims
        self.lambda_ = lambda_
        self.drop_prob = drop_prob

        # number of time steps or length of longest sequences/training example
        self.n_dense_layers = len(dense_layers_dims)
        self.normalize = normalize

        # instantiate layers
        self.character_lookup = Embedding(n_unique, emb_dim, name='character-lookup', embeddings_regularizer=L2(lambda_))
        self.lstm_layer = LSTM(units=n_a, return_sequences=True, return_state=True, name='lstm-layer')
        self.dense_layers = [Dense(units=dim, name=f'dense-layer-{i}', kernel_regularizer=L2(lambda_)) for i, dim, in enumerate(dense_layers_dims)]
        self.norm_layers = [BatchNormalization(name=f'norm-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
        self.act_layers = [Activation(activation=tf.nn.relu, name=f'act-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
        self.drop_layers = [Dropout(drop_prob, name=f'drop-layer-{i}') for i in range(len(dense_layers_dims) - 1)]

    def call(self, inputs, h=None, c=None, return_state=False, training=False):
        """
        args:
            inputs - 
            states - is a list containing initial state or current state
            h and c
            return_state -
            training - 
        """

        # get batch of training examples
        X = inputs
        embeddings = self.character_lookup(X, training=training)

        # check if states are empty. Note h and c will of
        # course be None during training so this is the reason
        # why self.lstm_layer.get_initial_state() is called
        # but during inference state will be provided using loop
        if h is None and c is None:
            h, c = self.lstm_layer.get_initial_state(embeddings)
        hs, h, c = self.lstm_layer(embeddings, initial_state=[h, c], training=training)

        temp = hs
        for i in range(self.n_dense_layers - 1):
            temp = self.dense_layers[i](temp, training=training)
                
            # if normalize is false do not permit passing temp 
            # to batch normalization layer
            if self.normalize == True:
                temp = self.norm_layers[i](temp, training=training)

            # note model only passes the activation to dropout 
            # during training
            temp = self.act_layers[i](temp, training=training)
            temp = self.drop_layers[i](temp, training=training)

        # output logits will be a (m, Ty, n_unique) 
        logits = self.dense_layers[-1](temp, training=training)

        # only is the logits, h, and c are returned during 
        # sampling but during training only logits are 
        # returned per batch of inputs
        return (logits, h, c) if return_state == True else logits
    
    def get_config(self):
        config = super(GenPhiloTextA, self).get_config()
        config['emb_dim'] = self.emb_dim
        config['n_a'] = self.n_a
        config['n_unique'] = self.n_unique
        config['dense_layers_dims'] = self.dense_layers_dims
        config['lambda_'] = self.lambda_
        config['drop_prob'] = self.drop_prob
        config['normalize'] = self.normalize

        return config


@tf.keras.utils.register_keras_serializable()
class GenPhiloTextB(tf.keras.Model):
    def __init__(self, emb_dim=32, n_a=128, n_unique=26, T_x=50, dense_layers_dims=[26], lambda_=1, drop_prob=0.0, normalize=False, **kwargs):
        super(GenPhiloTextB, self).__init__(**kwargs)
        self.emb_dim = emb_dim
        self.n_a = n_a
        self.n_unique = n_unique
        self.dense_layers_dims = dense_layers_dims
        self.lambda_ = lambda_
        self.drop_prob = drop_prob

        # number of time steps or length of longest sequences/training example
        self.n_time_steps = T_x
        self.n_dense_layers = len(dense_layers_dims)
        self.normalize = normalize

        # instantiate layers
        self.char_emb_layer = Embedding(n_unique, emb_dim, name='char-emb-layer', embeddings_regularizer=L2(lambda_))
        self.lstm_cell = LSTM(units=n_a, return_state=True, name='lstm-cell')
        self.dense_layers = [Dense(units=dim, name=f'dense-layer-{i}', kernel_regularizer=L2(lambda_)) for i, dim, in enumerate(dense_layers_dims)]
        self.norm_layers = [BatchNormalization(name=f'norm-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
        self.act_layers = [Activation(activation=tf.nn.relu, name=f'act-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
        self.drop_layers = [Dropout(drop_prob, name=f'drop-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
        
        # utility layers
        self.reshape_layer = Reshape(target_shape=(1, emb_dim), name='reshape-layer')

    def call(self, inputs, **kwargs):
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

            # because each timestep takes in a (m, 1, n_features)
            # input we must reshape our input x at timestep t
            # from (m, n_features) to (m, 1, n_features)
            x_t = self.reshape_layer(x_t)

            # pass the input x to the LSTM cell as well as the 
            # hidden and cell states that will constantly change
            states = self.lstm_cell(inputs=x_t, initial_state=[h, c])
            _, h, c = states

            # pass final hidden state to n dense, norm, act, and dropout layers
            temp = h
            for i in range(self.n_dense_layers - 1):
                temp = self.dense_layers[i](temp)
                
                # if normalize is false do not permit passing temp 
                # to batch normalization layer
                if self.normalize == True:
                    temp = self.norm_layers[i](temp)

                temp = self.act_layers[i](temp)

                # only pass the activation to dropout during training
                if kwargs['training'] == True:
                    temp = self.drop_layers[i](temp)

            # pass final activation or dropout (if during training) 
            # to final dense layer
            out_logit = self.dense_layers[-1](temp)

            # when all outputs are collected this will 
            # have dimensionality (T_y, m, n_unique)
            out_logits.append(out_logit)

        # reshape the (T_y, m, n_unique) to (m, T_y, n_unique)
        # by using tf.transpose(out_logits, perm=[1, 0, 2])
        out_logits = tf.transpose(out_logits, perm=[1, 0, 2])

        return out_logits
    
    def get_config(self):
        config = super(GenPhiloTextB, self).get_config()
        config['emb_dim'] = self.emb_dim
        config['n_a'] = self.n_a
        config['n_unique'] = self.n_unique
        config['T_x'] = self.n_time_steps
        config['dense_layers_dims'] = self.dense_layers_dims
        config['lambda_'] = self.lambda_
        config['drop_prob'] = self.drop_prob
        config['normalize'] = self.normalize

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

    # (m, T_x)
    X = Input(shape=(T_x, ))

    # (m, T_x, n_unique)
    embeddings = Embedding(n_unique, emb_dim)(X)

    # (m, T_x, n_a)
    h1, h2, c = LSTM(units=n_a, return_sequences=True, return_state=True)(embeddings)

    # (m, n_unique)
    logits = Dense(units=n_unique)(h1)

    return Model(inputs=X, outputs=logits)

def load_alt_model_b(emb_dim=32, n_a=128, n_unique=26, T_x=50, dense_layers_dims=[26], lambda_=1, drop_prob=0.0, normalize=False):
    """
    args:
        emb_dim -
        n_a - 
        n_unique - 
        T_x -
    """
    n_dense_layers = len(dense_layers_dims)

    # define shape of batch of inputs including 
    # hidden and cell states
    X = Input(shape=(T_x,), name='X')
    h_0 = Input(shape=(n_a,), name='init-hidden-state')
    c_0 = Input(shape=(n_a,), name='init-cell-state')

    # pass input X inside embedding layer such that X which is (m, T_x) 
    # is transformed to (m, T_x, n_features) which the LSTM layer can accept
    embeddings = Embedding(n_unique, emb_dim, name='character-lookup', embeddings_regularizer=L2(lambda_))(X)

    # define reshaper, Lstm, dense, norm, and act layers here
    # since each layer will only be used once this is because we need
    # the weights of the layers at each time step to be the same and not
    # constantly changing
    reshape_layer = Reshape(target_shape=(1, emb_dim), name='reshape-layer')
    lstm_cell = LSTM(units=n_a, return_state=True, name='lstm-cell')
    dense_layers = [Dense(units=dim, name=f'dense-layer-{i}', kernel_regularizer=L2(lambda_)) for i, dim in enumerate(dense_layers_dims)]
    norm_layers = [BatchNormalization(name=f'norm-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
    act_layers = [Activation(activation=tf.nn.relu, name=f'act-layer-{i}') for i in range(len(dense_layers_dims) - 1)]
    drop_layers = [Dropout(drop_prob, name=f'drop-layer-{i}') for i in range(len(dense_layers_dims) - 1)]

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
        temp = h
        for i in range(n_dense_layers - 1):
            temp = dense_layers[i](temp)

            # if normalize is false permit passing temp 
            # to batch normalization layer
            if normalize == True:
                temp = norm_layers[i](temp)

            temp = act_layers[i](temp)
            temp = drop_layers[i](temp)
            
        out_logit = dense_layers[-1](temp)

        out_logits.append(out_logit)

    return Model(inputs=[X, h_0, c_0], outputs=out_logits)

def generate(model, prompts: list, char_to_idx=None, T_x: int=250, chars_to_skip: list=['[UNK]'], temperature: float=1.0):
    """
    args:
        model - the model that was trained

        prompts - a list of string/s of any varying length e.g.
        ["Romeo", "Hello World", "Mike", "Charlie", "I", ...] that 
        will be used as seed to generate oen character per timestep
        using the trained model

        char_to_idx - the lookup layer/table to map characters to
        their respective indeces or id's

        T_x - desired length of the generated text

        chars_to_skip - a list representing the characters to skip so
        that the model is prevented from generating any character that is 
        a part of this list. Default is obviously the '[UNK]' character

        temperature - a value that represents how much diversity should
        the generative models text be allowed to have. A higher temp
        would mean more creativity or diversity of characters generated.
        A lower one would mean the opposite.
    """

    # get number of all unique chars including '[UNK]' char
    n_chars = len(char_to_idx.get_vocabulary())

    # ids to skip has shape (1, 1)
    ids_to_skip = char_to_idx(chars_to_skip)[:, None]
    sparse_mask_vector = tf.SparseTensor(values=[float('-inf')] * len(ids_to_skip), indices=ids_to_skip, dense_shape=[n_chars])
    dense_mask_vector = tf.reshape(tf.sparse.to_dense(sparse_mask_vector), shape=(1, -1))

    # initialize states and transform seed e.g.
    # "ROMEO" into a (1, T_x) input in this case
    # (1, 5) input [[<id of R>, <id of O>, ...]]
    input_chars = tf.strings.unicode_split(prompts, 'UTF-8')
    input_ids = char_to_idx(input_chars).to_tensor()
    h, c = None, None
    output_ids = input_ids
    print(output_ids)

    for _ in range(T_x):
        # calculate logits
        pred_logits, h, c = model(inputs=input_ids, h=h, c=c, return_state=True)
        
        # get only the last predicted char of the model, since 
        # seed will always vary e.g. a (1, 120) or (m, 120) will predict 
        # a (1, 120, n_unique) or (m, 120, n_unique) Y, likewise a
        # (1, 5) or (m, 5) will predict a (1, 5, n_unique) (m, 5, n_unique)
        # Y. SO get only the prediction at the last timestep. Output shape 
        # will now be (1, n_unique) or (m, n_unique).
        pred_logits = pred_logits[:, -1, :]
        pred_logits = pred_logits / temperature
        pred_logits = pred_logits + dense_mask_vector

        # sample from multinomial distribution
        pred_ids = tf.random.categorical(pred_logits, num_samples=1)

        # update output ids by concatenating newly predicted
        # set of ids, which will have shapes (m, T_x ) or (1, T_x) 
        # depending on number of given prompts
        output_ids = tf.concat([output_ids, pred_ids], axis=1)

        # set input_ids to the sampled and predicted ids and
        # start loop again for sampling
        input_ids = pred_ids

        if _ % 10 == 0:
            print(f"iteration {_}")
            print(f"predicted logits: {pred_logits}")

    # return all concatenated ids
    return output_ids

def load_inf_model_b(char_emb_layer, lstm_cell, dense_layers: list, norm_layers: list=None, char_to_idx=None, T_x: int=100, chars_to_skip: list=['[UNK]'], temperature: float=1.0):
    """
    args:
        char_emb_layer - 
        lstm_cell - 
        dense_layer - 
        norm_layer - 
    """

    # retrieve number of unique chars from dense layer
    n_dense_layers = len(dense_layers)
    n_chars = dense_layers[-1].units
    n_a = lstm_cell.units
    print(n_chars)
    print(n_a)

    # ids to skip has shape (1, 1)
    ids_to_skip = char_to_idx(chars_to_skip)[:, None]
    sparse_mask_vector = tf.SparseTensor(values=[float('-inf')] * len(ids_to_skip), indices=ids_to_skip, dense_shape=[n_chars])
    dense_mask_vector = tf.reshape(tf.sparse.to_dense(sparse_mask_vector), shape=(1, -1))
    print(dense_mask_vector)

    # declare also an add layer for adding the mask 
    # vector to the predicted logits
    add_layer = Add(name='add-layer')

    act_layers = [Activation(activation=tf.nn.relu, name=f'act-layer-{i}') for i in range(len(dense_layers) - 1)]

    # and after the add layer pass the masked logits 
    # to the activation layer
    out_layer = Activation(activation=tf.nn.softmax, name=f'out-layer')

    # add reshape layer after the using tf.argmax 
    # for the activation values
    reshape_layer = Reshape(target_shape=(1,), name='reshape-layer')

    # define shape of batch of inputs including hidden and cell 
    # states. Note in the prediction stage X will only be a (1, 1)
    # input representing one example and 1 timestep
    x_1 = Input(shape=(1,))
    h_0 = Input(shape=(n_a,), name='init_hidden_state')
    c_0 = Input(shape=(n_a,), name='init_cell_state')

    # a flag that represents how many newlines we have left before
    # sequence generation stops
    num_newlines = 2

    # assign hidden and cell states
    x_t = x_1
    h = h_0
    c = c_0

    print(T_x)
    # will store all (1, 1) predicted ids
    output_ids = []
    index = 0
    while index < T_x:
        # extract learned embeddings from embedding matrix.
        # once (1, 1) input is fed output embeddings will now be 
        # (1, 1, 32) 32 can be variable depending on the initially 
        # set number of features during training
        embedding = char_emb_layer(x_t)

        # since embedding x_t is already a (1, 1, 32) input
        # we can feed it directly to our lstm_cell
        states = lstm_cell(inputs=embedding, initial_state=[h, c])
        _, h, c = states

        # pass the hidden state to the dense 
        # layer and then normalize after
        temp = h
        for i in range(n_dense_layers - 1):
            temp = dense_layers[i](temp)

            if norm_layers != None:
                temp = norm_layers[i](temp)
                
            temp = act_layers[i](temp)

        z_t = dense_layers[-1](temp)    

        # because tensor after norm layer is (1, 57) for example 
        # should our n unique chars be 57, we must also have our
        # mask tensor to be of the same shape
        z_t = add_layer([z_t, dense_mask_vector]) / temperature

        # pass the the final logits to the activation 
        # which have output shape (1, 57)
        out = out_layer(z_t)

        # # this uses a greedy sample but may not be ideal in
        # # generating diverse characters in the trained model
        # pred_id = tf.argmax(out, axis=1)

        # # since after argmax the output shape will be (1,)
        # # denoting one example with 1 id we can reshape it to be (1, 1)
        # # in order for us to pass it again to the embedding layer
        # pred_id = reshape_layer(pred_id)

        # alternative sampler uses the multinomial distribution
        # to sample the probability values we have with slight
        # randomness this returns a (1, 1) sampled index of 
        # the character
        pred_id = tf.random.categorical(out, num_samples=1)

        # re assign x_t to newly predicted id to pass 
        # in next timestep
        x_t = pred_id
        
        # append predicted id to output array
        output_ids.append(pred_id)

        # increment index to sample for next timestep
        print(index)
        index += 1

    return Model(inputs=[x_1, h_0, c_0], outputs=output_ids)

        

if __name__ == "__main__":
    # hyperparameters
    m = 20000
    T_x = 100
    n_unique = 57
    n_a = 64
    emb_dim = 32
    dense_layers_dims = [n_unique]
    lambda_ = 0.8
    drop_prob = 0.4
    learning_rate = 1e-3
    epochs = 100
    batch_size = 512
    normalize = False

    # note X becomes (m, T_x, n_features) when fed to embedding layer
    X = np.random.randint(0, n_unique, size=(m, T_x))

    # we have to match the output of the prediction of our 
    # model which is a list of (100, 26) values. So instead of a 3D matrixc
    # we create a list fo 2D matrices of shape (100, 26)
    Y = [np.random.rand(m, n_unique) for _ in range(T_x)]

    # one hot encode our dummy (T_y, m, n_unique) probabilities
    Y = [tf.one_hot(tf.argmax(y, axis=1), depth=n_unique) for y in Y]
    
    # test for computing loss with (m, T_y, n_unique) predictions
    Y_true = tf.reshape(Y, shape=(m, T_x, n_unique))
    dummy_logits = np.random.randn(m, T_x, n_unique)
    loss = cce_loss(from_logits=True)(dummy_logits, Y_true)
    print(f"computed test loss: {loss}")

    # initialize hidden and cell states to shape (m, n_units)
    h_0 = np.zeros(shape=(m, n_a))
    c_0 = np.zeros(shape=(m, n_a))

    # instantiate custom model
    model = GenPhiloTextA(emb_dim=emb_dim, n_a=n_a, n_unique=n_unique, dense_layers_dims=dense_layers_dims, lambda_=lambda_, drop_prob=drop_prob, normalize=normalize)
    # model = GenPhiloTextB(emb_dim=emb_dim, n_a=n_a, n_unique=n_unique, T_x=T_x, dense_layers_dims=dense_layers_dims, lambda_=lambda_, drop_prob=drop_prob, normalize=normalize)

    # define loss, optimizer, and metrics then compile
    opt = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
    loss = cce_loss(from_logits=True)
    metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]    
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    model(X)
    # model([X, h_0, c_0])
    model.summary()

    # define checkpoint and early stopping callback to save
    # best weights at each epoch and to stop if there is no improvement
    # of validation loss for 10 consecutive epochs
    weights_path = f"../saved/weights/test_{model.name}" + "_{epoch:02d}_{val_loss:.4f}.h5"
    checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
    stopper = EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [checkpoint, stopper]

    # begin training test model
    history = model.fit(X, Y_true, 
        epochs=epochs,
        batch_size=batch_size, 
        callbacks=callbacks,
        validation_split=0.3,
        verbose=2,)
    
    # history = model.fit([X, h_0, c_0], Y_true, 
    #     epochs=epochs,
    #     batch_size=batch_size, 
    #     callbacks=callbacks,
    #     validation_split=0.3,
    #     verbose=2,)
    
    # save model
    # model.save_weights('../saved/weights/test_model_gen_philo_text.h5', save_format='h5')
    # model.save('../saved/models/test_model_b.h5', save_format='h5')
    