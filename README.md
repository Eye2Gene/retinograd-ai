# Retinograd ai
Gradeability model for FAF images

# PyTorch Classification Model Inference

This repository contains code for running inference of gradeability classification model.

## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/Eye2Gene/retinograd-ai.git
   cd retinograd-ai

2. Install the required packages
    ```sh
    pip install -r requirements.txt

3. Running Inference
    ```sh
    python inference.py --image_path path_to_image --csv_column column_of_csv_of_input_image_path --model_path path_to_model --output_csv_path path_to_output_csv

Predicted Labels:
1. Class 0: Ungradable
2. Class 1: Gradable
