import os
import math
import numpy as np
import tensorflow.keras as tfk


class SnapshotEnsembleCallback(tfk.callbacks.Callback):
    """
    Train TensorFlow Keras models with Cosine Annealing and save an ensemble of models with
    no additional computational expense.
    
    At the start of each annealing cycle, we compute the length of the new cycle
    and update learning rate bounds. For each epoch, we decay the learning rate
    via cosine annealing such that the maximum learning rate is achieved at
    the start of each cycle and the minimum achieved at the end. The parameters
    `cycle_length_multiplier` and `lr_multiplier` allow for dynamic annealing where 
    we may wish to anneal over a longer period and/or with lower learning rates
    as training progresses. The helper function `VisualizeLR()` may be used to visualize
    the learning rate schedule. See Loshchilov and Hunter (2017) for more details on
    cosine annealing.

    We may also save the model at the end of each annealing cycle in order to generate
    an ensemble of trained models (presumably near local minima of the loss surface) with
    no additional computational cost. See Huang et al. (2017).


    Parameters
    ----------
        cycle_length                (int)       Initial number of epochs per cycle
        cycle_length_multiplier     (float)     Multiplier on number of epochs per cycle
        lr_init                     (float)     Initial maximum learning rate
        lr_min                      (float)     Initial minimum learning rate
        lr_multiplier               (float)     Multiplier on learning rates per cycle
        ensemble                    (bool)      Whether to save an ensemble of models
        ensemble_options:           (dict)      Optional configuration:
            num_snapshots           (int)           Number of saved models
            dirpath                 (str)           Path to directory for saving models
            model_prefix            (str)           Prefix of model filenames
    
    References
    ----------
        Huang, G., Li, Y., & Pleiss, G. (2017). Snapshot Ensembles: Train 1, Get M for Free. 
            International Conference on Learning Representations. https://doi.org/https://doi.org/10.48550/arXiv.1704.00109
        Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. 
            International Conference on Learning Representations. https://doi.org/https://doi.org/10.48550/arXiv.1608.03983
    """

    def __init__(self, cycle_length=10, cycle_length_multiplier=1.5, lr_init=0.01, lr_min=1e-6, lr_multiplier=0.9,
                       ensemble=True, ensemble_options={}):
        super(SnapshotEnsembleCallback, self).__init__()
        self.cycle_length = cycle_length
        self.cycle_length_multiplier = cycle_length_multiplier
        self.lr_max = lr_init
        self.lr_min = lr_min
        self.lr_multiplier = lr_multiplier

        self.ensemble = ensemble
        ensembleConfig = { 
                            'num_snapshots' : 10, 
                            'dirpath' : 'Ensemble/', 
                            'model_prefix' : 'Model' 
                        }
        ensembleConfig.update( ensemble_options )
        self.ensembleConfig = ensembleConfig
        if self.ensemble:
            dirpath = self.ensembleConfig.get('dirpath')
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            print(f"Saving ensembled models to {dirpath}.")
        self.modelCounter = 0

        self.numEpochsTrained = 0.
        self.prevCycleEnd = 0.

    def on_batch_begin(self, batch_idx, logs={}):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('The optimizer does not have well-defined learning rates.')

        numBatches = self.params.get('steps')
        if numBatches == 0:
            raise Exception("Invalid number of batches.")

        # Update states at start of each annealing cycle
        t = self.numEpochsTrained + ( batch_idx / numBatches )
        if t >= (self.prevCycleEnd + self.cycle_length):
            self.cycle_length = self.cycle_length * self.cycle_length_multiplier
            self.lr_max = self.lr_max * self.lr_multiplier
            self.lr_min = self.lr_min * self.lr_multiplier
            self.prevCycleEnd = t

            if self.ensemble:
                self._SaveModel()

        # Cosine annealed learning rate
        cycleElapsed = t - self.prevCycleEnd
        lr = float(
                self.lr_min + 
                0.5 * (self.lr_max - self.lr_min) * 
                ( 1 + np.cos(np.pi * (cycleElapsed / self.cycle_length)) )
            )
        tfk.backend.set_value(self.model.optimizer.lr, lr)
        return

    def on_epoch_end(self, epoch_idx, logs={}):
        self.numEpochsTrained += 1
        return

    def _SaveModel(self):
        """
        Saves the model at the end of each annealing cycle.
        """

        # Reset model index to limit to K most recent snapshots
        model_idx = self.modelCounter + 1
        if model_idx > self.ensembleConfig.get('num_snapshots'):
            model_idx = 1
            self.modelCounter = 0

        # Save the model
        filepath = os.path.join(self.ensembleConfig.get('dirpath'), f"{self.ensembleConfig.get('model_prefix')}-{model_idx}.h5")
        self.model.save_weights(filepath, overwrite=True)

        self.modelCounter += 1

        return


    