# Image Denoising using Autoencoder

![Python](https://skillicons.dev/icons?i=python) ![TensorFlow](https://skillicons.dev/icons?i=tensorflow) ![Keras](https://skillicons.dev/icons?i=keras) ![NumPy](https://skillicons.dev/icons?i=numpy) ![Matplotlib](https://skillicons.dev/icons?i=matplotlib)

**Course:** Deep Learning
**Dataset:** Pistachio (Kirmizi) Images  
**Year:** 2025  



## About This Project
This project focuses on denoising images using deep learning autoencoders. Since the dataset lacked noisy images, Gaussian noise (mean=0.0, std=0.1) was added to generate noisy data. The project includes EDA, preprocessing, baseline and modified autoencoder models, and evaluation using **Structural Similarity Index (SSIM).**



## Dataset Features
The dataset consists of pistachio (kirmizi) images. Key preprocessing steps include:
- Resizing images to 100x100  
- Normalization and scaling  
- Generating synthetic noisy images using Gaussian noise (mean=0.0, std=0.1)  



## Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Visualized sample images and pixel distributions  
- Checked image resolutions, aspect ratios, and variability  
- Analyzed information to justify preprocessing and noise generation steps  

### 2. Data Preprocessing
- Split dataset into Train (80%), Validation (10%), and Test (10%)  
- Resized all images to 100x100 pixels  
- Added Gaussian noise to create synthetic noisy images  

### 3. Baseline Autoencoder Model
- Convolutional Autoencoder with kernel size 3x3  
- ReLU activation for hidden layers, Sigmoid for output layer  
- Optimizer: Adam  
- Loss function: MSE (Mean Squared Error)  
- Trained to reconstruct clean images from noisy inputs  

### 4. Modified Autoencoder Model
- Adjusted encoder/decoder architecture (e.g., more layers, different filter sizes, additional Conv layers)  
- Fine-tuned hyperparameters to improve denoising performance  
- Trained modified model and compared with baseline  

### 5. Evaluation
- Evaluated baseline and modified models using **SSIM (Structural Similarity Index)**  
- Analyzed reconstructed images and compared SSIM scores to assess denoising quality



## Technologies Used
- Python  
- TensorFlow & Keras  
- NumPy & pandas  
- Matplotlib & Seaborn  
- OpenCV / PIL (for image preprocessing)  
- scikit-image (for SSIM evaluation)

