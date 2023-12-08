# main model training here
from models.arcs import load_inf_model, GenPhiloText
from utilities.preprocessors import preprocess, get_chars, map_value_to_index, init_sequences_b, decode_predictions
from utilities.loaders import load_file
from utilities.visualizers import export_results

from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy as cce_metric

from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    try:
        # instantiate parser to take args from user in command line
        parser = ArgumentParser()
        parser.add_argument('--emb_dim', type=int, default=32, help='number of features to use in character embedding matrix/lookup')
        parser.add_argument('-n_a', type=int, default=128, help='number of units in an LSTM cell')
        parser.add_argument('-T_x', type=int, default=50, help='length (+ 1) of each partitioned sequence in the corpus')
        parser.add_argument('--dense_layers_dims', nargs='+', type=int, default=[64], help='number of layers and number of nodes in each dense layers of the language model')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
        parser.add_argument('--alpha', type=float, default=1e-4, help='learning rate of optimizers')
        parser.add_argument('--n_epochs', type=int, default=300, help='the number of epochs')

        args = parser.parse_args()
        print(args.dense_layers_dims)
        print(args.T_x)

        # load data and pass through preprocessing pipeline
        corpus = load_file('./data/notes.txt')
        corpus = preprocess(corpus)
        chars = get_chars(corpus)
        char_to_idx = map_value_to_index(chars)
        idx_to_char = map_value_to_index(chars, inverted=True)
        print("loading corpus successful!\n")

        n_unique = len(char_to_idx.get_vocabulary())

        # create dataset X and Y which will have shapes (m, T_x) 
        # and (T_y, m, n_unique) respectively
        
        
        X, Y = init_sequences_b(corpus, char_to_idx, T_x=args.T_x)    
        Y = [tf.one_hot(y, depth=n_unique) for y in tf.reshape(Y, shape=(-1, Y.shape[0]))]

        # get also number of examples created in init_sequences_b()
        # and initialize hidden and cell states to shape (m, n_units)
        m = X.shape[0]
        h_0 = np.zeros(shape=(m, args.n_a))
        c_0 = np.zeros(shape=(m, args.n_a))
        
        print("sequence creation successful")

        # define sample inputs and load model
        sample_input = tf.random.uniform(shape=(1, args.T_x), minval=0, maxval=n_unique - 1, dtype=tf.int32)
        sample_h = tf.zeros(shape=(1, args.n_a))
        sample_c = tf.zeros(shape=(1, args.n_a))
        model = GenPhiloText(emb_dim=args.emb_dim, n_a=args.n_a, n_unique=n_unique, T_x=args.T_x, dense_layers_dims=args.dense_layers_dims + [n_unique])
        model([sample_input, sample_h, sample_c])
        print(model.summary(), end='\n')

        # define loss, optimizer, and metrics and compile
        opt = Adam(learning_rate=args.alpha, beta_1=0.9, beta_2=0.999)
        loss = cce_loss(from_logits=True)
        metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        # define checkpoint callback to save best weights at each epoch
        weights_path = "./saved/weights/gen_philo_text_{epoch:02d}.h5"
        checkpoint = ModelCheckpoint(weights_path, monitor='categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        callbacks = [checkpoint]

        # being training model
        print("commencing model training...\n")
        history = model.fit([X, h_0, c_0], Y, 
            epochs=args.n_epochs, 
            batch_size=args.batch_size, 
            callbacks=callbacks)
        
        # export png iamge of results
        export_results(history, ['loss'], image_only=False)
        export_results(history, ['categorical_accuracy'], image_only=False)


    except ValueError as e:
        print(e)
        print("You have entered an unpermitted value for the number of timesteps T_x. Try a higher value")