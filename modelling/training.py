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

from argparse import ArgumentParser

if __name__ == "__main__":
    try:
        # instantiate parser to take args from user in command line
        parser = ArgumentParser()
        parser.add_argument('--emb_dim', type=int, default=32, help='number of features to use in character embedding matrix/lookup')
        parser.add_argument('--n_a', type=int, default=128, help='number of units in an LSTM cell')
        parser.add_argument('-T_x', type=int, default=50, help='length (+ 1) of each partitioned sequence in the corpus')
        parser.add_argument('--dense_layers_dims', nargs='+', type=int, default=[26], help='number of layers and number of nodes in each dense layers of the language model')
        parser.add_argument('--batch_size', type=int, default=128, help='batch size during training')
        parser.add_argument('--alpha', type=float, default=1e-4, help='learning rate of optimizers')
        parser.add_argument('--n_epochs', type=int, default=300, help='the number of epochs')

        args = parser.parse_args()

        # load data and pass through preprocessing pipeline
        corpus = load_file('./data/notes.txt')
        corpus = preprocess(corpus)
        chars = get_chars(corpus)
        char_to_idx = map_value_to_index(chars)
        char_to_idx = map_value_to_index(chars, inverted=True)

        n_unique = len(char_to_idx.get_vocabulary())

        # create dataset X and Y which will have shapes (m, T_x) 
        # and (T_y, m, n_unique) respectively
        X, Y = init_sequences_b(corpus, char_to_idx, T_x=args.T_x)
        Y = [tf.onehot(y, depth=n_unique) for y in tf.reshape(Y, shape=(-1, Y.shape[0]))]

        # define sample inputs and load model
        sample_input = tf.random.uniform(shape=(1, 50), minval=0, maxval=n_unique - 1, dtype=tf.int32)
        sample_h = tf.zeros(shape=(1, 128))
        sample_c = tf.zeros(shape=(1, 128))
        model = GenPhiloText(emb_dim=args.emb_dim, n_a=args.n_a, n_unique=n_unique, T_x=args.T_x, dense_layers_dims=[64, n_unique])
        model([sample_input, sample_h, sample_c])
        model.summary()

        # define loss, optimizer, and metrics
        opt = Adam(learning_rate=args.alpha, beta_1=0.9, beta_2=0.999)
        loss = cce_loss(from_logits=True)
        metrics = [CategoricalAccuracy(), cce_metric(from_logits=True)]

        # define checkpoint callback to save best weights at each epoch
        weights_path = "./saved/weights/gen_philo_text_{epoch:02d}_{categorical_accuracy:.4f}.hdf5"
        checkpoint = ModelCheckpoint(weights_path, monitor='categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
        callbacks = [checkpoint]

        # being training model
        history = model.fit(X, Y, 
            epochs=args.n_epochs, 
            batch_size=args.batch_size, 
            callback=callbacks)
        
        # export png iamge of results
        export_results(history, ['loss'], image_only=False)
        export_results(history, ['categorical_accuracy'], image_only=False)


    except ValueError as e:
        print("You have entered an unpermitted value for the number of timesteps T_x. Try a higher value")