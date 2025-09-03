import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import argparse
import pandas as pd
import os
import timm
from tqdm import tqdm


def load_model(model_path):
    try:
        model = timm.create_model('inception_resnet_v2', pretrained=True,num_classes=2)
        print(model_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

def predict(model, image_path):
    preprocess = transforms.Compose([
    transforms.Resize((768, 768)),
    transforms.ToTensor(),
])

    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: File not found - {image_path}")
        return None
    except UnidentifiedImageError:
        print(f"Error: Cannot identify image file - {image_path}")
        return None

    image = preprocess(image)
    image = image.unsqueeze(0)

    try:
        with torch.no_grad():
            output = model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)
        return predicted.item()
    except Exception as e:
        print(f"Error during prediction for image {image_path}: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--csv_column', type=str,default='image_path', required=True, help='Column of input image path of CSV file')
    parser.add_argument('--model_path', type=str, default='weights/model.pth', help='Path to the model weights')
    parser.add_argument('--output_csv_path', type=str, default='predictions.csv', help='Path to save the output CSV file')
    args = parser.parse_args()

    model = load_model(args.model_path)
    if model is None:
        print("Model could not be loaded. Exiting.")
        exit(1)

    try:
        df = pd.read_csv(args.csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found - {args.csv_path}")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty - {args.csv_path}")
        exit(1)
    except pd.errors.ParserError:
        print(f"Error: CSV file is corrupt - {args.csv_path}")
        exit(1)

    if args.csv_column not in df.columns:
        print(f"Error: CSV file does not contain '{args.csv_column}' column")
        exit(1)

    predictions = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        image_path = row[args.csv_column]
        prediction = predict(model, image_path)
        predictions.append(prediction)
    
    df['predictions'] = predictions
    df.to_csv(args.output_csv_path, index=False)

    print(f'Predictions saved to {args.output_csv_path}')
