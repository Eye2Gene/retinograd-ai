# Retinograd ai
Gradeability model for FAF images
This repository contains a Python script for performing image classification using a pre-trained Inception ResNet v2 model specifically designed to assess the gradeability of Fundus Autofluorescence (FAF) images. The script reads image paths from a CSV file, processes each image, and outputs the gradeability predictions to a new CSV file. This tool is intended to assist researchers and clinicians in evaluating the quality and diagnostic utility of FAF images.

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

3. Download model weights

    Go to the [Releases]([https://github.com/yourusername/image-classification/releases](https://github.com/Eye2Gene/retinograd-ai/releases/) page of this repository. Under the latest release, find the `FAF_inception_resenet_2Class_classification.pth` file listed under "Assets" and download it to the `weights` directory in your project.

4. Running Inference
    ```sh
    python classify_images.py --csv_path <input_csv> --csv_column <image_path_column> --model_path <model_weights> --output_csv_path <output_csv>

Arguments
--csv_path (str): Path to the input CSV file containing image paths.
--csv_column (str): Column name in the CSV file that contains the image paths. Default is image_path.
--model_path (str): Path to the model weights file. Model weight file has been added in releases.
--output_csv_path (str): Path to save the output CSV file with predictions. Default is predictions.csv.

The input csv should have a column of input image paths and that should given after input argument --csv_column.

Predicted Labels:
1. Class 0: Ungradable
2. Class 1: Gradable


   ```sh
   #Example
   python classify_images.py --csv_path data/images.csv --csv_column image_path --model_path weights/model.pth --output_csv_path data/predictions.csv

##License

This project is licensed under the MIT License. See the LICENSE file for details.

##Cite

To be added soon...
