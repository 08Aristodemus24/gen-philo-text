# define data visualizer functions here
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl

import numpy as np



def export_results(history, metrics_to_use=['loss', 
                                            'val_loss', 
                                            'binary_crossentropy', 
                                            'val_binary_crossentropy', 
                                            'binary_accuracy', 
                                            'val_binary_accuracy', 
                                            'precision', 
                                            'val_precision', 
                                            'recall', 
                                            'val_recall', 
                                            'f1_m', 
                                            'val_f1_m', 
                                            'auc', 
                                            'val_auc',
                                            'categorical_crossentropy',
                                            'val_categorical_crossentropy'], image_only=False):
    for index in range(0, len(metrics_to_use) - 1, 2):
        # >>> list(range(0, 5, 2))
        # [0, 2, 4]
        metrics_indeces = (index, index + 1)

        view_results(
            results=build_results(
                history,
                metrics=(metrics_to_use[index], metrics_to_use[index + 1])),

            curr_metrics_indeces=metrics_indeces,
            epochs=history.epoch, 
            image_only=image_only,
            img_title="generative models performance {}".format(
                metrics_to_use[index],
            )
        )

def build_results(history, metrics: list):
    """
    builds the dictionary of results based on metric history of both models

    args:
        history - the history object returned by the self.fit() method of the
        tensorflow Model object

        metrics - a list of strings of all the metrics to extract and place in
        the dictionary
    """
    results = {}
    for metric in metrics:
        if metric not in results:
            results[metric] = history.history[metric]

    return results

def view_results(results: dict, epochs: list, curr_metrics_indeces: tuple, image_only: bool, img_title: str='figure'):
    """
    plots the number of epochs against the cost given cost values across these epochs
    """

    # use matplotlib backend
    mpl.use('Agg')

    figure = plt.figure(figsize=(15, 10))
    axis = figure.add_subplot()

    styles = [
        ('p:', '#f54949'), 
        ('h-', '#f59a45'), 
        ('o--', '#afb809'), 
        ('x:','#51ad00'), 
        ('+:', '#03a65d'), 
        ('8-', '#035aa6'), 
        ('.--', '#03078a'), 
        ('>:', '#6902e6'),
        ('p-', '#c005e6'),
        ('h--', '#fa69a3'),
        ('o:', '#240511'),
        ('x-', '#052224'),
        ('+--', '#402708'),
        ('8:', '#000000')]

    for index, (key, value) in enumerate(results.items()):
        if key == "loss" or key == "val_loss":
            # e.g. loss, val_loss has indeces 0 and 1
            # binary_cross_entropy, val_binary_cross_entropy 
            # has indeces 2 and 3
            axis.plot(np.arange(len(epochs)), value, styles[curr_metrics_indeces[index]][0], color=styles[curr_metrics_indeces[index]][1], alpha=0.5, label=key)
        else:
            metric_perc = [round(val * 100, 2) for val in value]
            axis.plot(np.arange(len(epochs)), metric_perc, styles[curr_metrics_indeces[index]][0], color=styles[curr_metrics_indeces[index]][1], alpha=0.5, label=key)

    # annotate end of lines
    for index, (key, value) in enumerate(results.items()):        
        if key == "loss" or key == "val_loss":
            last_loss_rounded = round(value[-1], 2)
            axis.annotate(last_loss_rounded, xy=(epochs[-1], value[-1]), color=styles[curr_metrics_indeces[index]][1])
        else: 
            last_metric_perc = round(value[-1] * 100, 2)
            axis.annotate(last_metric_perc, xy=(epochs[-1], value[-1] * 100), color=styles[curr_metrics_indeces[index]][1])

    axis.set_ylabel('metric value')
    axis.set_xlabel('epochs')
    axis.set_title(img_title)
    axis.legend()

    plt.savefig(f'./figures & images/{img_title}.png')

    if image_only is False:
        plt.show()

    # delete figure
    del figure