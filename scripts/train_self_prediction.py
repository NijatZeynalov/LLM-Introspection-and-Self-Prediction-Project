from llm_introspection.config import Config
import torch
from llm_introspection.self_prediction_model import SelfPredictionModel

def main():
    config = Config()
    self_prediction_model = SelfPredictionModel(config)

    # Prepare the self-prediction task data
    prompts = [
        "What would be the second word in my response to this prompt?",
        "Would I choose a long-term or short-term option in this scenario?",
        "How confident am I that my response will be helpful?"
    ]
    target_outputs = [
        "second",
        "long-term",
        "confident"
    ]

    # Fine-tune the model on the self-prediction task
    self_prediction_model.fine_tune_on_self_prediction(prompts, target_outputs)

    # Save the fine-tuned model
    torch.save(self_prediction_model.model.state_dict(), f"{config.output_dir}/self_prediction_model.pth")

if __name__ == "__main__":
    main()