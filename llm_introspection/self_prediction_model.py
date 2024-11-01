import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
import logging

logging.basicConfig(level=logging.INFO)

class SelfPredictionModel:
    """
    Represents the self-prediction model based on the LLaMA 2 7B language model.
    """
    def __init__(self, config):
        self.config = config
        try:
            self.model = LlamaForCausalLM.from_pretrained(config.llm_model_name)
            self.tokenizer = LlamaTokenizer.from_pretrained(config.llm_model_name)
        except Exception as e:
            logging.error(f"Error loading model or tokenizer: {e}")
            raise

        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)

    def fine_tune_on_self_prediction(self, prompts, target_outputs):
        """
        Fine-tune the model on the self-prediction task using mini-batches.
        """
        self.model.train()
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            for prompt, target in zip(prompts, target_outputs):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                labels = self.tokenizer.encode(target, return_tensors="pt").to(self.device)
                loss = self.compute_loss(input_ids, labels)
                epoch_loss += loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            logging.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(prompts)}")
            torch.cuda.empty_cache()

    def compute_loss(self, input_ids, labels):
        """
        Compute the loss for the self-prediction task.
        """
        outputs = self.model(input_ids, labels=labels)
        return outputs.loss

    def predict_self_behavior(self, prompt):
        """
        Predict the model's own behavior given a prompt.
        """
        self.model.eval()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(
            input_ids=input_ids,
            max_length=100,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            num_beams=2,
            early_stopping=True,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
