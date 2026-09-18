In this project,I have built a deep learning model to check if a given currency note is counterfeit or not by ResNet50 model and later integrated Grad-CAM visualize discriminative regions influencing classification decisions.
Counterfeit money is illegally produced currency designed to imitate real money and deceive people. Counterfeit currency significantly destabilises a nation's economy & erodes public trust in its financial systems. When fake money enters circulation, it shifts the financial burden to ordinary citizens, small businesses & national security infrastructure. Counterfeit networks are masterminded by a spectrum of criminals, ranging from state-sponsored crime syndicates executing economic warfare to tech-savvy local gangs exploiting modern consumer technology. 
Hence, deep-learning can be used for automated image classification and check for fake currency notes.
The dataset used for this model is 'Indian Currency Real vs Fake Notes Dataset' on Kaggle.This dataset contains high-quality images of Indian currency notes, categorized as real or fake across six denominations — ₹10, ₹20, ₹50, ₹100, ₹500, and ₹2000.

Indian Currency Real vs Fake Notes Dataset/
│
├── real/
│   ├── 10/
│   ├── 20/
│   ├── 50/
│   ├── 100/
│   ├── 500/
│   └── 2000/
│
└── fake/
    ├── 10/
    ├── 20/
    ├── 50/
    ├── 100/
    ├── 500/
    └── 2000/
Each folder contains real and fake currency note images for that denomination.

Here, I have used Resnet50 to perform a binary classification on this dataset. ResNet-50 is a 50-layer deep Convolutional Neural Network (CNN) used mainly for computer vision & image recognitions.
Resnet50 serves as a baseline feature extractor for further features like object detection & image segmentation.
Resnet50 has 50 weight layers, balancing strong accuracy & training speed. 
Deep neural network often suffers from vanishing gradient issues. Vanishing gradient happens during backpropagation when the model finds gradients by chain rule, multiplying partial derivatives across multiple layers. If the derivative values are less than one, repeated multiplication causes the gradient to shrink exponentially as it moves backward. Weights in the initial layers get tiny updates or none at all, leaving the layers effectively frozen. Resnet50 solves this issue by using shortcut connections to pass input data directly to later layers.
Resnet50 also learns small changes (residuals) rather than full target mappings, making deep optimization straightforward. 

The Resnet50 model gave this final results:

Test Accuracy: 0.9930
Test F1 Score: 0.9947

Classification Report:
              precision    recall  f1-score   support

        fake       0.99      0.99      0.99       626
        real       0.99      1.00      0.99      1235

    accuracy                           0.99      1861
   macro avg       0.99      0.99      0.99      1861
weighted avg       0.99      0.99      0.99      1861

After Resnet50, I have used Gradcam to show which parts of an image Resnet50 looked at to make a prediction.
The ResNet50 model classified the input banknote as genuine with a predicted probability of 95.7%

