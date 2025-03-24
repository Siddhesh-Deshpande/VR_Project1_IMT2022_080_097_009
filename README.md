# Introduction
This project aims to classify and segment face masks in images. The techniques used are:
- ML Classifiers (Random Forest, SVM, Neural Network)
- Convolutional Neural Network for classification
- Segmentation using traditional techniques (Edge detection, Thresholding)
- U-Net for Mask Segmentation

The methodology for each task is mentioned in the report provided, along with the code. Instructions for setup and execution are given below.

Team Member: 
- Siddhesh Deshpande: IMT2022080
- Madhav Girdhar: IMT2022009
- Krish Patel: IMT2022097

## Task-1 & 2 : Binray Classification using Handcarft Feature and ML classifier & Binray Classification using CNN.

This section focuses on classifying face mask usage in images using machine learning classifiers. Feature extraction is performed using Histogram of Oriented Gradients (HOG), which captures important edge and texture details. The extracted features are then used to train various classifiers, including Support Vector Machine (SVM), Random Forest, and Neural Network, to accurately distinguish between masked and unmasked faces.Additionally, a Convolutional Neural Network (CNN) is used for automatic feature extraction and classification. 

## Dependencies

To run the project you need follwing python libraries:
- OpenCv (cv2)
- Numpy
- Scikit-learn (sklearn)
- TensorFlow
- Scikit-image (skimage.feature)
- torch
- torchvision

Make sure all these are present on your system or else install them using following command: 
```bash
pip install opencv-python numpy scikit-learn scikit-image tensorflow torch torchvision
```

## To Run Program
- Make sure the dataset is downloaded; otherwise, download it from the following link: https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset. This dataset contains images of people with and without face masks for classification tasks.
- Adjust the data loading path in your script to correctly point to the dataset directory.
- Now, run the notebook to obtain the results for both tasks.

## Results

- Support Vector Machine : ```Accuracy: 0.9426```
- Random Forest : ```Accuracy: 0.9130```
- Neural Network ```Accuracy: 0.9377```
- Convolution Neural Network: ```Accuracy: 0.9799```

| Models                | Precision(Class-0) | Precision(Class-1) | Recall (Class-0)  | Recall (Class-1)|
| -                     |   -               |    -               |  -                 |  -              |
| SUPPORT VECTOR MACHINE|   0.93            |       0.95         |       0.94         |    0.94         |
| RANDOM FOREST         |   0.92            |       0.89         |       0.86         |    0.94         | 
| NEURAL NETWORK        |   0.90            |       0.94         |       0.93         |    0.92          |
| CONVOLUTION NEURAL NETWORK |              |                    |                    |                 |


## Task-3 & 4 : Region Segmentation Using Traditional Techniques & Mask Segmentation Using U-Net

This section focuses on segmenting mask regions using both traditional and deep learning methods. Region-based techniques like thresholding or edge detection are applied for initial segmentation. A U-Net model is then trained for precise mask segmentation. The results from both methods are evaluated using IoU and Dice score. Finally, their performance is compared to determine the most effective approach.

## Dependencies

To run the project you need follwing python libraries:
- OpenCv (cv2)
- Numpy
- TensorFlow
- matplotlib

Make sure all these are present on your system or else install them using following command: 
```bash
pip install opencv-python numpy tensorflow matplotlib
```

## To Run Program
- Make sure the dataset is downloaded; otherwise, download it from the following link: https://github.com/sadjadrz/MFSD. It contains the image as well as ground truth face masks in the form of binary images.
- Adjust the data loading path in your script to correctly point to the dataset directory.
- Now, run the notebook to obtain the results for both tasks.

## Results
