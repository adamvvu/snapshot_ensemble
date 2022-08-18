from snapshot_ensemble import *
import matplotlib.pyplot as plt

def VisualizeLR(epochs=500, cycle_length=10, cycle_length_multiplier=1.5, lr_init=0.01, lr_min=1e-6, lr_multiplier=0.9):
    """
    Helper function for visualizing the cosine annealed learning rate schedule.

    Note: The implementation in `SnapshotEnsembleCallback` allows for smoother batch-level decay of learning rates, while
    this function returns a simplified epoch-level decay for visualization purposes.
    """
    lr_max = lr_init
    prevRestartEpoch = 0
    res = []
    for t in range(epochs):
        
        # Update states at start of new cycle
        if t == (prevRestartEpoch + cycle_length + 1):
            cycle_length = math.ceil(cycle_length * cycle_length_multiplier)
            lr_max = lr_max * lr_multiplier
            lr_min = lr_min * lr_multiplier
            prevRestartEpoch = t
            
        # Cosine annealed learning rate
        epochs_since_restart = t - prevRestartEpoch
        lr = float(
                lr_min + 
                0.5 * (lr_max - lr_min) * 
                ( 1 + np.cos(np.pi * (epochs_since_restart / cycle_length)) )
            )
        res.append( lr )

    fig = plt.figure()
    plt.title('Cosine Annealed Learning Rate Schedule')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    _ = plt.plot(res)
    plt.close()
    return fig


def GenerateSnapshotCallbacks(cycle_length=10, cycle_length_multiplier=1.5, lr_init=0.01, lr_min=1e-6, lr_multiplier=0.9,
                              ensemble=True, ensemble_options={}):
    """
    Helper function for generating a list of Keras callbacks to be used during training.

    Includes the `SnapshotEnsembleCallback` for cosine annealing and saving an ensemble of models,
    as well as `ModelCheckpoint` for saving the best model.
    """

    snapEns = SnapshotEnsembleCallback(
                                cycle_length=cycle_length, 
                                cycle_length_multiplier=cycle_length_multiplier, 
                                lr_init=lr_init, 
                                lr_min=lr_min, 
                                lr_multiplier=lr_multiplier,
                                ensemble=ensemble, 
                                ensemble_options=ensemble_options
                            )

    callbacks = [
        snapEns,
        tfk.callbacks.ModelCheckpoint(
                                    os.path.join( snapEns.ensembleConfig.get('dirpath'), f"{snapEns.ensembleConfig.get('model_prefix')}-Best.h5" ),
                                    monitor='val_loss',
                                    mode='min',
                                    save_best_only=True, 
                                    save_weights_only=True
                                ),
    ]

    return callbacks
