# **DEVELOPMENT FINISHED, DEPLOYMENT PENDING DUE TO EXCEEDING FILE SIZE**

# A Model that will generate novel sequences of philosophical text based on writings about Jungian psychology, Biblical philosophy, and the lot. Built with React.js, Flask, and Tensorflow

# Usage usage
1. assuming git is installed clone repository by running `git clone https://github.com/08Aristodemus24/gen-philo-text`
2. assuming conda is also installed run `conda create -n <environment name e.g. gen-philo-text> python=3.11.5`. Note python version should be `3.11.5` for the to be created conda environment to avoid dependency/package incompatibility.
3. run `conda activate <environment name used>` or `activate <environment name used>`.
4. run `conda list -e` to see list of installed packages. If pip is not yet installed run conda install pip, otherwise skip this step and move to step 5.
5. navigate to `gen-philo-text/server-side` folder using `cd gen-philo-text/server-side`.
5. run `pip install -r requirements.txt` inside the directory we navigated to which was `/gen-philo-text/server-side` which contains the `requirements.txt` file
6. after installing packages/dependencies run `python index.py` while in this directory to run app locally
7. in browser go to `http://127.0.0.1:5000/`
8. control panel of app will have 3 inputs: prompt, temperature, and sequence length. Prompt can be understood as the starting point in which our model will append certain words during generation for instance if the prompt given is "jordan" then model might generate "jordan is a country in the middle east" and so on. Temperature input can be understood as "how much the do you want the model to generate diverse sequences or words?" e.g. if a diversity of 2 (this is the max value for diversity/temperature by the way) then then the model might potentially generate incomprehensible words (almost made up words) e.g. "jordan djanna sounlava kianpo". And lastly Sequence Length is how long do you want the generated sequence to be in terms of character length for isntance if sequence length is 10 then generated sequence would be "jordan is."

# Model Building
**To do:**

**Articles:**

**Problems:**
1. subclassing model giving different results to using Model API. https://stackoverflow.com/questions/65851897/subclassing-of-model-class-and-model-functional-api-give-different-results-in-te. Could it be that using hte functional API is better than by using subclassing? What if all along I could've used the former in my previous recommender system project?
2. there is a reason why accuracy can be so low in generative text models, it's because other metrics like perplexity can be used to evaluate whether or not our models performs well. https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
3. maybe why the model generates such (albeit novel, but nevertheless) shit words and sentences is because of how I partititioned the corpus.

**Insights:**
1. https://datascience.stackexchange.com/questions/97132/my-lstm-has-a-really-low-accuracy-is-there-anyway-to-improve-it:

Your model is overfitting. There are a couple ways to deal with this:
* The data you have is just 5k entries large. This is not suitable for Deep Learning algos which require a lot more data than this. You would be wise to use a ML algorithm for this data. Overfitting happens when the model you use is quite complicated for the data you have. Either use a ML algo or gather some more data if you want to use DL algos.
* You are using accuracy as a metric for a multiclass classification problem which is not advised. Try using a proper metric and see if it solves the problem.
* Another way to tackle this is as @niv dudovitch suggests. Reduce the complexity of the LSTM by reducing the no of parameters.
* You can try using Keras Tuner to tune the HP and see if it solves the problem.
* Try suing Batch Normalization instead of Dropout. If you really want to use Dropout then using a value of 0.5 is too large a value for such a small data.

solution: in text generation models you can use other metrics other than accuracy such as bleu score or perplexity

2. test model finally predicted more diverse characters although too diverse and without coherence, is it merely because of the random numbers the sampler uses, the temperature, or because the model has low quality data and has underfitted?

* https://stackoverflow.com/questions/46924452/what-to-do-when-seq2seq-network-repeats-words-over-and-over-in-output/59407187#59407187
* https://stackoverflow.com/questions/60605838/using-rnn-for-text-generation-it-always-predicts-the-same-letter/60764713#60764713



# Running model training
**To do:**
1. training parameters
python training.py -d notes --emb_dim 32 -n_a 64 -T_x 100 --dense_layers_dims 64 32 --batch_size 128 --alpha 1e-3 --lambda_ 0.8 --drop_prob 0.4 --n_epochs 100

python training.py -d notes --emb_dim 256 -n_a 512 -T_x 100 --batch_size 128 --alpha 1e-3 --lambda_ 0.8 --n_epochs 30

python training.py -d notes --emb_dim 256 -n_a 512 -T_x 100 --batch_size 128 --alpha 1e-3 --lambda_ 0.8 --n_epochs 100


# Setting up backend
**To do:**
1. install flask, flask-cors
2. write model loader code so that upon running backend server, written code will run before making a request
3. run flask server by `flask --app <name of app e.g. server.py> run`
