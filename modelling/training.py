# main model training here
from models.arcs import load_inf_model, GenPhiloText
from utilities.preprocessors import preprocess, get_chars, map_value_to_index, init_sequences_b, decode_predictions
from utilities.loaders import load_file
from utilities.visualizers import export_results

from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy as cce_metric

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

import numpy as np
from argparse import ArgumentParser

if __name__ == "__main__":
    try:
        # instantiate parser to take args from user in command line
        parser = ArgumentParser()
        parser.add_argument('-d', type=str, default="shakespeare", help="what text dataset/corpus to train model on")
        parser.add_argument('--emb_dim', type=int, default=32, help='number of features to use in character embedding matrix/lookup')
        parser.add_argument('-n_a', type=int, default=64, help='number of units in an LSTM cell')
        parser.add_argument('-T_x', type=int, default=100, help='length (+ 1) of each partitioned sequence in the corpus')
        parser.add_argument('--dense_layers_dims', nargs='+', type=int, default=[], help='number of layers and number of nodes in each dense layers of the language model')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
        parser.add_argument('--alpha', type=float, default=1e-3, help='learning rate of optimizers')
        parser.add_argument('--lambda_', type=float, default=0.8, help='regularization constant during training')
        parser.add_argument('--drop_prob', type=float, default=0.4, help='percentage at which to drop nodes before next dense layer')
        parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
        parser.add_argument('--normalize', type=bool, default=False, help='whether to use batch normalization after each dense layer')
        args = parser.parse_args()

        # load data and pass through preprocessing pipeline
        corpus = load_file(f'./data/{args.d}.txt')
        corpus = preprocess(corpus)
        chars = get_chars(corpus)
        char_to_idx = map_value_to_index(chars)
        idx_to_char = map_value_to_index(chars, inverted=True)
        print("loading corpus successful!\n")

        n_unique = len(char_to_idx.get_vocabulary())

        # create dataset X and Y which will have shapes (m, T_x) 
        # and (m, T_y, n_unique) respectively
        X, Y = init_sequences_b(corpus, char_to_idx, T_x=args.T_x)
        Y = tf.transpose(tf.one_hot(Y, depth=n_unique, axis=1), perm=[0, 2, 1])
        # print(Y)

        # get also number of examples created in init_sequences_b()
        # and initialize hidden and cell states to shape (m, n_units)
        m = X.shape[0]
        h_0 = np.zeros(shape=(m, args.n_a))
        c_0 = np.zeros(shape=(m, args.n_a))
        print(f"number of examples: {m}")
        print("sequence creation successful")

        # define sample inputs and load model
        sample_input = tf.random.uniform(shape=(1, args.T_x), minval=0, maxval=n_unique - 1, dtype=tf.int32)
        sample_h = tf.zeros(shape=(1, args.n_a))
        sample_c = tf.zeros(shape=(1, args.n_a))
        model = GenPhiloText(
            emb_dim=args.emb_dim, 
            n_a=args.n_a, 
            n_unique=n_unique, 
            T_x=args.T_x, 
            dense_layers_dims=args.dense_layers_dims + [n_unique], 
            lambda_=args.lambda_, 
            drop_prob=args.drop_prob,
            normalize=args.normalize)
        model([sample_input, sample_h, sample_c])
        print(model.summary(), end='\n')

        # define loss, optimizer, and metrics and compile
        opt = Adam(learning_rate=args.alpha, beta_1=0.9, beta_2=0.999)
        loss = cce_loss(from_logits=True)
        metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        # define checkpoint and early stopping callback to save
        # best weights at each epoch and to stop if there is no improvement
        # of validation loss for 10 consecutive epochs
        weights_path = f"./saved/weights/{args.d}_gen_philo_text" + "_{epoch:02d}_{val_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        stopper = EarlyStopping(monitor='val_loss', patience=10)
        callbacks = [checkpoint, stopper]

        # being training model
        print("commencing model training...\n")
        history = model.fit([X, h_0, c_0], Y, 
            epochs=args.n_epochs, 
            batch_size=args.batch_size, 
            callbacks=callbacks,
            validation_split=0.3,
            verbose=2)
        
        # export png iamge of results
        export_results(history, args.d, ['loss', 'val_loss'], image_only=False)


    except ValueError as e:
        print(e)
        print("You have entered an unpermitted value for the number of timesteps T_x. Try a higher value")