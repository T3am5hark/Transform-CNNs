# Utilities for working with Keras neural networks
import matplotlib.pyplot as plt
import numpy as np


def describe_model(model):
    '''
    describe_model(model)

    inputs:
        model : Keras sequential neural network
    returns:
        None
    Description:
        Iterate the layers in the model and output relevant structural
        metadata.
    '''
    for idx,layer in enumerate(model.layers):
        print('Layer {0}: {1}'.format(idx, layer.__class__))
        print('    input={0}\n    output={1}'.format(layer.input_shape, layer.output_shape))
        if hasattr(layer, 'activation'):
            activation = layer.activation
            print('    act={0}'.format(activation))
        if hasattr(layer, 'rate'):
            rate = layer.rate
            print('    rate={0}'.format(rate))
        if hasattr(layer, 'strides'):
            print('    strides={0}'.format(layer.strides))


def hits_and_misses(predictions, targets):
    '''
    (hits, misses) = hits_and_misses(predictions, targets)

    inputs:
        predictions - vector of class indices, pre-collapsed via argmax
        targets -     target class indices

    returns:
        hits, misses - arrays of sample indices indicating correctly- vs
                       incorrectly-classified data points.
    '''
    if len(predictions.shape) > 1:
        predictions=predictions.argmax(axis=1)
    if len(targets.shape) > 1:
        targets=targets.argmax(axis=1)
    missed = targets != predictions
    misses = [index for index,val in enumerate(missed) if val == True]
    hits   = [index for index,val in enumerate(missed) if val == False]
    return hits, misses

def plot_training_data(model):
    '''
    plot_training_data(model)

    description:
        Plots the loss and accuracy metrics for train and test sets over the
        epochs of training for the NNET.

    inputs:
        model - a Keras model that has been trained on a dataset
    '''
    n_epochs = len(model.history["loss"])
    x_data = np.arange(0, n_epochs)
    plt.figure(figsize=(8,5))
    plt.plot(x_data, model.history["loss"], marker='.', label="train_loss")
    plt.plot(x_data, model.history["val_loss"], marker='.', label="val_loss")
    plt.plot(x_data, model.history["acc"], marker='.', label="train_acc")
    plt.plot(x_data, model.history["val_acc"], marker='.', label="val_acc")
    plt.title('Training Loss & Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.legend()
    plt.grid()
    plt.show()
