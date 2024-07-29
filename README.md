# Retinograd ai

Gradeability model for FAF images.

This repository contains a Python script for performing image classification using a pre-trained Inception ResNet v2 model specifically designed to assess the gradeability of Fundus Autofluorescence (FAF) images. The script reads image paths from a CSV file, processes each image, and outputs the gradeability predictions to a new CSV file. This tool is intended to assist researchers and clinicians in evaluating the quality and diagnostic utility of FAF images.

> [!note]
>
> If you use Retinograd in your work, please cite us as follows:
>
> Retinograd-AI: An Open-source Automated Gradeability Assessment for Retinal Scans for Inherited Eye Dystrophies <br>
> URL: Available soon! <br>
> <sub><sup>William Woof*, Gunjan Naik*, Saoud Al-Khuzaei*, Thales Antonio Cabral De Guimaraes, Malena Daich Varela, Sagnik Sen, Pallavi Bagga, Ismail Moghul, Michel Michaelides, Konstantinos Balaskas, Nikolas Pontikos</sub></sup>


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Eye2Gene/retinograd-ai.git
   cd retinograd-ai
   ```

2. Install the required packages
    ```bash
    pip install -r requirements.txt
    ```

3. Download model weights

    Go to the [Releases](https://github.com/Eye2Gene/retinograd-ai/releases/) page of this repository. Under the latest release, find the `FAF_inception_resenet_2Class_classification.pth` file listed under "Assets" and download it to the `weights` directory in your project.

   ```bash
   wget -O weights/model.pth https://github.com/Eye2Gene/retinograd-ai/releases/download/V0.0.1/FAF_inception_resenet_2Class_classification.pth
   ```

5. Running Inference
    ```bash
    python classify_images.py --csv_path <input_csv> --csv_column <image_path_column> --model_path <model_weights> --output_csv_path <output_csv>

    # For example:
    python classify_images.py --csv_path data/images.csv --csv_column image_path --model_path weights/model.pth --output_csv_path data/predictions.csv
    ```

## Arguments
- `--csv_path` (str): Path to the input CSV file containing image paths.
- `--csv_column` (str): Column name in the CSV file that contains the image paths. Default is `image_path`.
- `--model_path` (str): Path to the model weights file. Model weight file has been added in releases.
- `--output_csv_path` (str): Path to save the output CSV file with predictions. Default is `predictions.csv`.

The input CSV requires at least one column containing all the paths to the input images. The name of this column should be passed using `--csv_column <col_name>`.

Retinograd-ai currently predicted two labels for each image:

1. Class 0: Ungradable
2. Class 1: Gradable

## License

This project is licensed under the MIT License. See the [LICENSE file](https://github.com/Eye2Gene/retinograd-ai/blob/main/LICENSE) for details.
