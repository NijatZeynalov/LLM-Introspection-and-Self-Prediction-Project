import torch
from .self_prediction_model import SelfPredictionModel

class IntrospectionEvaluator:
    """
    Evaluates the model's introspection capability.
    """
    def __init__(self, config, self_prediction_model):
        self.config = config
        self.self_prediction_model = self_prediction_model

    def evaluate_introspection(self, prompts, expected_outputs):
        """
        Evaluate the model's self-prediction accuracy.
        """
        accuracy = 0.0
        for prompt, expected in zip(prompts, expected_outputs):
            predicted = self.self_prediction_model.predict_self_behavior(prompt)
            if predicted == expected:
                accuracy += 1
        return accuracy / len(prompts)

    def evaluate_cross_model_prediction(self, prompts, target_model):
        """
        Evaluate the cross-model prediction accuracy.
        """
        accuracy = 0.0
        for prompt in prompts:
            predicted = self.self_prediction_model.predict_self_behavior(prompt)
            actual = target_model.predict_self_behavior(prompt)
            if predicted == actual:
                accuracy += 1
        return accuracy / len(prompts)