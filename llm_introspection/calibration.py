import torch
import numpy as np

class CalibrationEvaluator:
    """
    Evaluates the model's calibration in self-prediction tasks.
    """
    def __init__(self, config, self_prediction_model):
        self.config = config
        self.self_prediction_model = self_prediction_model

    def evaluate_calibration(self, prompts):
        """
        Evaluate the model's calibration in self-prediction tasks.
        """
        calibration_scores = []
        for prompt in prompts:
            predicted_output = self.self_prediction_model.predict_self_behavior(prompt)
            predicted_probs = self.compute_predicted_probs(prompt, predicted_output)
            calibration_score = self.compute_calibration_score(predicted_probs, predicted_output)
            calibration_scores.append(calibration_score)
        return np.mean(calibration_scores)

    def compute_predicted_probs(self, prompt, predicted_output):
        """
        Compute the predicted probabilities for the generated output.
        """
        input_ids = self.self_prediction_model.tokenizer.encode(prompt, return_tensors="pt").to(self.self_prediction_model.device)
        output_ids = self.self_prediction_model.tokenizer.encode(predicted_output, return_tensors="pt").to(self.self_prediction_model.device)
        logits = self.self_prediction_model.model(input_ids)[0]
        probs = torch.softmax(logits, dim=-1)
        return probs[0, :-1, output_ids[0]]

    def compute_calibration_score(self, predicted_probs, predicted_output):
        """
        Compute the calibration score for the self-predicted output.
        """
        # Compute the expected confidence (average of predicted probabilities)
        expected_confidence = predicted_probs.mean().item()

        # Compute the actual accuracy (1 if the prediction is correct, 0 otherwise)
        actual_accuracy = 1.0 if self.self_prediction_model.tokenizer.decode(predicted_output[0]) == predicted_output else 0.0

        # Compute the calibration score as the absolute difference between expected confidence and actual accuracy
        calibration_score = np.abs(expected_confidence - actual_accuracy)

        return 1 - calibration_score