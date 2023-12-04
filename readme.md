# **STILL IN DEVELOPMENT**

# A Model that will generate novel sequences of philosophical text based on writings about Jungian psychology, Biblical philosophy, and the lot.

# Model Building
**To do:**

**Articles:**

**Problems:**
1. subclassing model giving different results to using Model API. https://stackoverflow.com/questions/65851897/subclassing-of-model-class-and-model-functional-api-give-different-results-in-te. Could it be that using hte functional API is better than by using subclassing? What if all along I could've used the former in my previous recommender system project?
2. there is a reason why accuracy can be so low in generative text models, it's because other metrics like perplexity can be used to evaluate whether or not our models performs well. https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1

**Insights:**
1. https://datascience.stackexchange.com/questions/97132/my-lstm-has-a-really-low-accuracy-is-there-anyway-to-improve-it:

Your model is overfitting. There are a couple ways to deal with this:
* The data you have is just 5k entries large. This is not suitable for Deep Learning algos which require a lot more data than this. You would be wise to use a ML algorithm for this data. Overfitting happens when the model you use is quite complicated for the data you have. Either use a ML algo or gather some more data if you want to use DL algos.
* You are using accuracy as a metric for a multiclass classification problem which is not advised. Try using a proper metric and see if it solves the problem.
* Another way to tackle this is as @niv dudovitch suggests. Reduce the complexity of the LSTM by reducing the no of parameters.
* You can try using Keras Tuner to tune the HP and see if it solves the problem.
* Try suing Batch Normalization instead of Dropout. If you really want to use Dropout then using a value of 0.5 is too large a value for such a small data.

solution: in text generation models you can use other metrics other than accuracy such as bleu score or perplexity
2. 