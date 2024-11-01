# LLM Introspection and Self-Prediction Project

## Overview
This project aims to train a language model (Llama 2 7b used) to predict its own behavior and responses. 

## Features
- **Self-Prediction**: The model is trained to predict what it might say when given a prompt, essentially trying to understand its own responses.
- **Introspection Evaluation**: The project evaluates how well the model can "think about" its own behavior, measuring how accurate its self-predictions are.
- **Calibration**: The model also checks how confident it is in its own predictions, aiming to match high confidence with high accuracy.

## How to Use
1. **Set Up**: Install the required dependencies from `requirements.txt`.
2. **Configure**: Modify the configuration in `config.py` or use a custom configuration file.
3. **Train the Model**: Use the training script to fine-tune the model on the self-prediction tasks.
4. **Evaluate**: Run the evaluation scripts to assess how well the model predicts its own behavior and how calibrated its responses are.

## Installation
1. Clone this repository.
2. Install dependencies using:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Scripts
- **Training**: Use `train_self_prediction.py` to train the model.
- **Evaluating Introspection**: Run `evaluate_introspection.py` to measure the model's accuracy.
- **Cross-Model Comparison**: Use `cross_model_prediction.py` to compare predictions between two models.


