import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.style.use('ggplot')

fontP = FontProperties()
fontP.set_size('large')

palette = sns.diverging_palette(220, 20, sep=1, n=256)


def _value_to_color(val, color_min, color_max):
    val_position = 1 - float((val - color_min)) / (color_max - color_min)
    ind = int(val_position * 255)
    return palette[ind]


def plot_performances(training_instances, plot_title, fitness_name, save_location=None):

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12, 8))
    plt.title(plot_title, y=2.3)

    min_acc = min(training_instance.get_performances().loc[("val", "accuracy")].iloc[-1] for training_instance in training_instances)
    max_acc = max(training_instance.get_performances().loc[("val", "accuracy")].iloc[-1] for training_instance in training_instances)

    for training_instance in training_instances:

        performances = training_instance.get_performances()

        color = _value_to_color(performances.loc[("val","accuracy")].iloc[-1], color_min=min_acc, color_max=max_acc)

        ax[0].plot(np.arange(0, training_instance.epochs), performances.loc[("train", "accuracy")], color=color)
        ax[0].plot(np.arange(0, training_instance.epochs), performances.loc[("val", "accuracy")], linestyle='--', color=color)

        ax[1].plot(np.arange(0, training_instance.epochs), performances.loc[("train", "fitness")], color=color, label=training_instance.instance_name)
        ax[1].plot(np.arange(0, training_instance.epochs), performances.loc[("val", "fitness")], linestyle='--', color=color)

    ax[0].set_xticks([])
    ax[0].set_ylabel("Accuracy")

    ax[1].set_ylabel(fitness_name)
    ax[1].set_xlabel("Epoch")

    ax[1].legend(prop=fontP)

    if save_location is not None:
        plt.savefig(save_location)
