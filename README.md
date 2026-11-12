In this project,I have built a deep learning model to check if a given currency note is counterfeit or not by Residual Network architecture variants and later integrated Grad-CAM visualize discriminative regions influencing classification decisions on the best Resnet variant.

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

I have decided to apply 3 variants of Resnet: Resnet50,Resnet34 and Resnet101

ResNet-50 is a 50-layer deep Convolutional Neural Network (CNN) used mainly for computer vision & image recognitions.
Resnet50 serves as a baseline feature extractor for further features like object detection & image segmentation.
Resnet50 has 50 weight layers, balancing strong accuracy & training speed. 
Deep neural network often suffers from vanishing gradient issues. Vanishing gradient happens during backpropagation when the model finds gradients by chain rule, multiplying partial derivatives across multiple layers. If the derivative values are less than one, repeated multiplication causes the gradient to shrink exponentially as it moves backward. Weights in the initial layers get tiny updates or none at all, leaving the layers effectively frozen. Resnet50 solves this issue by using shortcut connections to pass input data directly to later layers.
Resnet50 also learns small changes (residuals) rather than full target mappings, making deep optimization straightforward. 

Resnet34 is a 34-layer deep cnn architecture(33 convolutional layers + 1 fully connected layer). It uses skip connections to bypass layers, allowing the network to learn residual mappings rather than direct transformations. Resnet34 uses the Basic Block configuration (that has two 3 × 3 convolutional layers with batch normalization and ReLU activations) rather than the heavy bottleneck blocks used in deeper variants like ResNet-50. 16 residual blocks are linked by identity shortcuts. Resnet34  Solves the vanishing gradient problem and degradation problem (where deeper networks unexpectedly suffer from higher training & test errors). Resnet34 gets high accuracy with fewer parameters compared to older deep networks like VGG-19, making gradient flow smooth across hundreds or thousands of theoretical layers.



