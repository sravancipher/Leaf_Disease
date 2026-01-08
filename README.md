# Plant Disease Segmentation with U-Net

This project implements a U-Net model for segmenting plant diseases from leaf images using TensorFlow and Keras.

## Prerequisites

- Python 3.8 or higher
- Git

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sravancipher/Leaf_Disease.git
   cd Leaf_Disease
   ```

2. Create a virtual environment:
   ```bash
   python -m venv dis_venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     dis_venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source dis_venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To train the U-Net model, run the training script:

```bash
python train/train_unet.py
```

The script will:
- Load the dataset (ensure PlantVillage images are available)
- Train the model for 25 epochs
- Save the trained model to `models/unet_leaf_segmentation.keras`

## Dataset

The model is trained on the PlantVillage dataset. Place the images in a `PlantVillage` folder and masks in `PlantVillage_Masks` (if available). Note: The current dataset may not have proper segmentation masks; for full segmentation training, use a dataset with paired images and masks.

## Model Architecture

- U-Net with encoder-decoder structure
- Input size: 572x572
- Loss: Combination of Dice loss and Binary Cross-Entropy
- Optimizer: Adam

## Files

- `main.py`: Basic model definition
- `models/unet.py`: U-Net implementation
- `train/train_unet.py`: Training script
- `utils/losses.py`: Custom loss functions
- `requirements.txt`: Dependencies

## Contributing

Feel free to submit issues or pull requests.

## License

This project is open-source.