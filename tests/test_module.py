from snapshot_ensemble import *
import pytest
import numpy as np
import glob
import os
np.random.seed(42)
import tensorflow.keras as tfk


def _GenerateData(N=100, numFeatures=3):

    X = np.random.random((N, numFeatures))

    beta = np.random.random(numFeatures)
    Y = np.expand_dims(np.dot(X, beta), axis=-1)
    return X, Y

def _CompileDNNModel(numFeatures=4, architecture=[6,6]):
    x = tfk.Input(shape=(numFeatures,))
    f = x
    for nodes in architecture:
        f = tfk.layers.Dense(nodes, activation='relu')(f)
    f = tfk.layers.Dense(1)(f)
    
    model = tfk.Model(inputs=x, outputs=f)
    model.compile(
        loss=tfk.losses.MeanSquaredError(),
        optimizer=tfk.optimizers.Adam(),
    )
    return model

def _TestStandardTrain(X, Y, architecture=[6,6]):

    numFeatures = X.shape[-1]

    model = _CompileDNNModel(numFeatures=numFeatures, architecture=architecture)

    model.fit(X, Y, epochs=10)
    model.predict(X)
    return

def _TestSnapshotEnsemble(X, Y, architecture=[6,6], dirpath='Ensemble/'):

    numFeatures = X.shape[-1]

    model = _CompileDNNModel(numFeatures=numFeatures, architecture=architecture)

    snapEns = SnapshotEnsembleCallback(cycle_length=1, ensemble_options={'dirpath':dirpath})

    model.fit(X, Y, epochs=10, callbacks=[snapEns])
    model.predict(X)

    savedModels = glob.glob(os.path.join(dirpath, '*.h5'))
    if len(savedModels) == 0:
        raise Exception('No saved models found.')

    models = []
    for file in savedModels:
        model = _CompileDNNModel(numFeatures=numFeatures, architecture=architecture)
        model.load_weights( file )
        models.append( model )

    Y_ens = []
    for model in models:
        y = model.predict(X)
        Y_ens.append( y )
    Y_ens = np.concatenate(Y_ens, axis=-1)

    return

def test_RunAllTests():

    N = 100
    numFeatures = 3
    architecture = [6,6]

    X, Y = _GenerateData(N=N, numFeatures=numFeatures)

    _TestStandardTrain(X=X, Y=Y, architecture=architecture)

    _TestSnapshotEnsemble(X=X, Y=Y, architecture=architecture)

    return