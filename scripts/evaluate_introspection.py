from llm_introspection.config import Config
from llm_introspection.self_prediction_model import SelfPredictionModel
from llm_introspection.evaluation import IntrospectionEvaluator
from llm_introspection.calibration import CalibrationEvaluator

def main():
    config = Config()
    self_prediction_model = SelfPredictionModel(config)
    self_prediction_model.model.load_state_dict(torch.load(f"{config.output_dir}/self_prediction_model.pth"))

    introspection_evaluator = IntrospectionEvaluator(config, self_prediction_model)
    calibration_evaluator = CalibrationEvaluator(config, self_prediction_model)

    # Evaluate the model's introspection capability
    prompts = [
        "What would be the second word in my response to this prompt?",
        "Would I choose a long-term or short-term option in this scenario?",
        "How confident am I that my response will be helpful?"
    ]
    expected_outputs = [
        "second",
        "long-term",
        "confident"
    ]
    introspection_accuracy = introspection_evaluator.evaluate_introspection(prompts, expected_outputs)
    print(f"Introspection Accuracy: {introspection_accuracy:.2f}")

    # Evaluate the model's calibration in self-prediction
    calibration_score = calibration_evaluator.evaluate_calibration(prompts)
    print(f"Calibration Score: {calibration_score:.2f}")

if __name__ == "__main__":
    main()