from llm_introspection.config import Config
from llm_introspection.self_prediction_model import SelfPredictionModel
from llm_introspection.evaluation import IntrospectionEvaluator

def main():
    config = Config()
    self_prediction_model = SelfPredictionModel(config)
    self_prediction_model.model.load_state_dict(torch.load(f"{config.output_dir}/self_prediction_model.pth"))

    # Load another LLaMA 2 7B model for cross-model prediction
    target_model = SelfPredictionModel(config)
    target_model.model.load_state_dict(torch.load(f"{config.output_dir}/target_model.pth"))

    introspection_evaluator = IntrospectionEvaluator(config, self_prediction_model)

    # Evaluate cross-model prediction accuracy
    prompts = [
        "What would be the second word in my response to this prompt?",
        "Would I choose a long-term or short-term option in this scenario?",
        "How confident am I that my response will be helpful?"
    ]
    cross_model_accuracy = introspection_evaluator.evaluate_cross_model_prediction(prompts, target_model)
    print(f"Cross-Model Prediction Accuracy: {cross_model_accuracy:.2f}")

if __name__ == "__main__":
    main()