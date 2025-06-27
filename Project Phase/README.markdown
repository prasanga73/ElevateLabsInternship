# Leaf Disease Classification Using PlantVillage Dataset

## Overview
This project implements a convolutional neural network (CNN) using PyTorch to classify leaf diseases from a subset of the [PlantVillage dataset](https://www.plantvillage.org/). The dataset covers 15 classes across Pepper, Potato, and Tomato plants. The model, optimized through hyperparameter tuning, leverages data augmentation, batch normalization, and dropout, achieving a test accuracy of 97.26% after 30 epochs with early stopping. This demonstrates strong potential for automated agricultural diagnostics.

## Dataset
The PlantVillage dataset subset is stored in `PlantDataset/train` and `PlantDataset/test`, with 15 classes:
- **Pepper**: Bacterial spot, Healthy
- **Potato**: Early blight, Late blight, Healthy
- **Tomato**: Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Tomato Yellow Leaf Curl Virus, Tomato mosaic virus, Healthy

### Preprocessing
- **Training**: Resized to 72x72, random crop to 64x64, random horizontal flips (p=0.5), rotations (15°), color jitter, Gaussian blur (p=0.2), normalization (mean=0.5, std=0.5).
- **Test**: Resized to 64x64, normalized (mean=0.5, std=0.5).
- Data loaded using `torchvision.datasets.ImageFolder` with a batch size of 32.

## Model Architecture
The `CustomCNN` model, defined in `tuningmodel.py`, consists of:
- **Conv Block 1**: Conv2d (3→32, 3x3), BatchNorm2d, ReLU, MaxPool2d (3x3), Dropout (p=0.2027).
- **Conv Block 2**: Two Conv2d (32→64, 64→64, 3x3), BatchNorm2d, ReLU, MaxPool2d (2x2), Dropout (p=0.2027).
- **Conv Block 3**: Two Conv2d (64→128, 128→128, 3x3), BatchNorm2d, ReLU, MaxPool2d (2x2), Dropout (p=0.2027).
- **Fully Connected**: Linear (Shape →1024), ReLU, BatchNorm1d, Dropout (p=0.3276), Linear (1024→15).
- Total parameters: ~3.57 million.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plantvillage-leaf-disease.git
   cd plantvillage-leaf-disease
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv pytorchenv
   source pytorchenv/bin/activate  # On Windows: pytorchenv\Scripts\activate
   pip install torch torchvision torchinfo matplotlib optuna
   ```
3. Download the PlantVillage dataset subset and place it in `PlantDataset/` with `train/` and `test/` subdirectories.

## Usage
1. **Run the Notebook**:
   - Open `main.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure the dataset is in `PlantDataset/`.
   - Execute cells to preprocess data, train the model, and plot results.
2. **Hyperparameter Tuning**:
   - Run `tuning.ipynb` to perform hyperparameter tuning using Optuna.
3. **Train the Model**:
   ```python
   from trainNN import train
   from tuningmodel import CustomCNN
   import torch
   from torch.utils.data import DataLoader
   from torchvision import datasets, transforms

   # Define transforms and data loaders (as in main.ipynb)
   train_loader = DataLoader(datasets.ImageFolder("PlantDataset/train", transform=train_transform), batch_size=32, shuffle=True)
   test_loader = DataLoader(datasets.ImageFolder("PlantDataset/test", transform=test_transform), batch_size=32, shuffle=True)

   # Initialize model, optimizer, loss function, and scheduler
   model = CustomCNN(num_classes=15, dropout_conv=0.2027, dropout_fc=0.3276).to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.000632, weight_decay=2.79e-06)
   criterion = torch.nn.CrossEntropyLoss()
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

   # Train
   results = train(model, train_loader, test_loader, optimizer, criterion, epochs=50, device=device, scheduler=scheduler)
   ```
4. **Save Model**:
   - The trained model is saved as `models/modeldrop_final.pth`.
5. **Visualize Results**:
   - Loss and accuracy curves are generated using `helper_functions.plot_loss_curves`.

## Results
- **Test Accuracy**: 97.26% (epoch 30, early stopping).
- **Test Loss**: 0.0908.
- **Training Time**: ~23.562 minutes on GPU.
- Loss and accuracy curves are saved as `loss_curves.png`.

## Project Structure
```
plantvillage-leaf-disease/
├── PlantDataset/
│   ├── train/              # Training images
│   ├── test/               # Test images
├── models/
│   ├── modeldrop_final.pth # Saved model weights
├── main.ipynb              # Main training notebook
├── tuning.ipynb            # Hyperparameter tuning notebook
├── tuningmodel.py          # CustomCNN definition
├── trainNN.py              # Training function
├── helper_functions.py     # Plotting utilities
└── README.md
```

## Future Work
- Deploy the model in a mobile app for real-time disease detection.
- Expand the dataset to include more plant species and diseases.
- Experiment with transfer learning using pre-trained models (e.g., ResNet).

## License
This project is licensed under the MIT License.