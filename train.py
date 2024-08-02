import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from torchvision import transforms
from tqdm import tqdm
from torch import nn, optim
import pandas as pd
import numpy as np
from PIL import Image
import timm
import os

class CustomDataset(Dataset):
    def __init__(self, csv_file, image_path_column,label_column,transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.image_path_column = image_path_column
        self.label_column = label_column
        

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.loc[idx, self.image_path_column]
        image = Image.open(img_name).convert('RGB')
        label = int(self.data_frame.loc[idx, self.label_column])
        
        label=1 if label>0 else 0

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(mode,train_csv, val_csv,image_path_column,label_column, model_path, num_epochs=25, batch_size=4, learning_rate=1e-5):
    transform = transforms.Compose([
        transforms.Resize((768, 768)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    
    train_dataset = CustomDataset(train_csv, image_path_column,label_column,transform=transform)
    val_dataset = CustomDataset(val_csv, image_path_column,label_column,transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=2)

    if mode == 'finetune':
        if os.path.isfile('./weights/model.pth'):
            print("Pretrained model file is present")
            checkpoint = torch.load('./weights/model.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded pre-trained weights.")
        else:
            print("Pretrained model file is not present in weights folder. Download it under weights folder")

    model = model.to(device)

    #Class weights for handling imbalance
    train_df=pd.read_csv(train_csv)
    train_df[label_column]=train_df[label_column].astype('int')
    
    print("Class weights :" +str(np.unique(train_df[label_column])))

    class_weights = compute_class_weight(class_weight='balanced',classes= np.unique(train_df[label_column]),y= train_df[label_column].values)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print(class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    torch.save({'model_state_dict': model.state_dict()}, model_path)

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'finetune'], required=True, help='Mode: train or finetune')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to the train CSV file')
    parser.add_argument('--val_csv', type=str, required=True, help='Path to the validation CSV file')
    parser.add_argument('--image_path_csv_column', type=str,default='image_path', required=True, help='Path to the validation CSV file')
    parser.add_argument('--label_csv_column', type=str, default='grade',required=True, help='Path to the validation CSV file')
    parser.add_argument('--model_path', type=str, default='weights/best_model.pth', help='Path to save the model weights')
    parser.add_argument('--num_epochs', type=int, default=25, help='Number of epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for the optimizer')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    trained_model = train_model(mode=args.mode,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        image_path_column=args.image_path_csv_column,
        label_column=args.label_csv_column,
        model_path=args.model_path,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
