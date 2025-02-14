import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod

class AbstractMethodMissingException(Exception):
    pass

class Algorithm(ABC):
    @abstractmethod
    def build_model():
        raise AbstractMethodMissingException("build model method needs to be implemented in the algorithm class!")
    
    @abstractmethod
    def learn():
        raise AbstractMethodMissingException("learn method needs to be implemented in the algorithm class!")
    
    def update_model():
        pass

    def before_train(self, task):
        pass
    
    def after_train(self, task):
        pass
    
    def before_evaluate(self, task):
        pass
    
    def after_evaluate(self, task):
        pass
    
    def copy_model_weights(self, source, target):
        ''' in-memory copy of model weights '''
        for source_layer, target_layer in zip(source.layers, target.layers):
            source_weights = source_layer.get_weights()
            target_layer.set_weights(source_weights)
            target_weights = target_layer.get_weights()

            if source_weights and all(tf.nest.map_structure(np.array_equal, source_weights, target_weights)):
                print(f'\033[93m[INFO]\033[0m [ALGO]: WEIGHT TRANSFER: {source.name}-{source_layer.name} -> {target.name}-{target_layer.name}')