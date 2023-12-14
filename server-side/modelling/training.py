# main model training here
from utilities.preprocessors import preprocess, get_chars, map_value_to_index, init_sequences_b, decode_predictions
from utilities.loaders import load_file, save_hyper_params, save_lookup_table
from utilities.visualizers import export_results

from tensorflow.keras.losses import CategoricalCrossentropy as cce_loss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import CategoricalAccuracy, CategoricalCrossentropy as cce_metric

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

import numpy as np
import json
from argparse import ArgumentParser


if __name__ == "__main__":
    try:
        # instantiate parser to take args from user in command line
        parser = ArgumentParser()
        parser.add_argument('-d', type=str, default="shakespeare", help="what text dataset/corpus to train model on")
        parser.add_argument('--model', type=str, default="GenPhiloTextA", help="what model to use")
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
        n_unique = len(char_to_idx.get_vocabulary())
        print("loading corpus successful!\n")

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

        if args.model.lower() == "genphilotexta":
            from models.arcs import GenPhiloTextA as Model

            # instantiate architecture of, build, and load model
            model = Model(
                emb_dim=args.emb_dim, 
                n_a=args.n_a, 
                n_unique=n_unique, 
                dense_layers_dims=args.dense_layers_dims + [n_unique], 
                lambda_=args.lambda_, 
                drop_prob=args.drop_prob,
                normalize=args.normalize)
            
            input = X
        elif args.model.lower() == "genphilotextb":
            from models.arcs import GenPhiloTextB as Model

            model = Model(
                emb_dim=args.emb_dim, 
                n_a=args.n_a, 
                n_unique=n_unique, 
                T_x=args.T_x, 
                dense_layers_dims=args.dense_layers_dims + [n_unique], 
                lambda_=args.lambda_, 
                drop_prob=args.drop_prob,
                normalize=args.normalize)
            
            input = [X, h_0, c_0]
        else:
            raise RuntimeError("Model chosen does not exist.")
            
        model(input)
        print(model.summary(), end='\n')
        
        # define loss, optimizer, and metrics and compile
        opt = Adam(learning_rate=args.alpha, beta_1=0.9, beta_2=0.999)
        loss = cce_loss(from_logits=True)
        metrics = [CategoricalAccuracy(), cce_metric(from_logits=True), 'accuracy']
        model.compile(loss=loss, optimizer=opt, metrics=metrics)

        # define checkpoint and early stopping callback to save
        # best weights at each epoch and to stop if there is no improvement
        # of validation loss for 10 consecutive epochs
        weights_path = f"./saved/weights/{args.d}_{model.name}" + "_{epoch:02d}_{val_loss:.4f}.h5"
        checkpoint = ModelCheckpoint(weights_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
        stopper = EarlyStopping(monitor='val_loss', patience=5)
        callbacks = [checkpoint, stopper]

        # being training model
        print("commencing model training...\n")
        history = model.fit(input, Y,
            epochs=args.n_epochs, 
            batch_size=args.batch_size, 
            callbacks=callbacks,
            validation_split=0.3,
            verbose=2)
        
        # export png iamge of results
        export_results(history, args.d, ['loss', 'val_loss'], image_only=False)
        export_results(history, args.d, ['categorical_crossentropy', 'val_categorical_crossentropy'], image_only=False)
        export_results(history, args.d, ['accuracy', 'val_accuracy'], image_only=False)

        # export all hyperparams used
        # converting args using vars() will result in the dicitonary
        # of all hyper parameters e.g. {'d': 'notes', 'model': 'GenPhiloTextA', 
        # 'emb_dim': 256, 'n_a': 512, 'T_x': 100, 'dense_layers_dims': [], 
        # 'batch_size': 128, 'alpha': 0.001, 'lambda_': 0.8, 'drop_prob': 0.4, 
        # 'n_epochs': 100, 'normalize': False}
        # Note to add n_unique to hyper params dict
        hyper_params = vars(args)
        hyper_params['n_unique'] = n_unique
        save_hyper_params('./saved/misc/hyper_params.json', hyper_params)

        # save also the lookup table used
        save_lookup_table('./saved/misc/char_to_idx', char_to_idx.get_vocabulary(include_special_tokens=False))

    except ValueError as e:
        print(e)

    except RuntimeError as e:
        print(e)