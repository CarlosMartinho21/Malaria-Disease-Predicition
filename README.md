# Malaria Disease Prediction Project

Malaria  is a severe illness caused by the Plasmodium parasite and transmitted through infected mosquitoes, that remains a significant global health concern. According to recent reports, it is estimated that there will be approximately 241 million cases and 627 thousand deaths worldwide by 2022, with concerns that the COVID-19 pandemic could exacerbate the situation. The World Health Organization (WHO) has implemented strategies to prevent, treat, and monitor malaria, emphasizing the importance of early diagnosis.

Traditional methods of malaria detection, such as microscopy and rapid diagnostic tests, are labor-intensive, subjective, and reliant on skilled personnel. To address these challenges, deep learning techniques can be seen like promising alternatives. These techniques aim to automate and improve the accuracy of malaria diagnosis.

Deep learning architectures, notably convolutional neural networks (CNNs), have gained traction in medical imaging analysis, including the detection of malaria parasites in blood smear images.


## Overview

This project aims to develop a machine learning model to predict the presence of malaria based on microscopic images of blood smears. Early and accurate diagnosis is crucial for effective treatment and disease control. By leveraging machine learning techniques, this project seeks to automate the process of malaria diagnosis, enabling faster and more accessible healthcare services. 

## Dataset

The dataset used in this project consists of microscopic images of blood smears collected from patients infected and uninfected with malaria parasites extracted from  Tensor Flow, a free and open-source software library for machine learning and artificial intelligence (https://www.tensorflow.org/datasets/catalog/malaria?hl=pt).
The malaria dataset comprises a total of 27,558 images of cells with equal instances of parasitized and non-infected cells from thin blood smear images of segmented cells.
Each image is labeled with the corresponding diagnosis (parasitized or non-infected).
The dataset only provides the training set, so it will be necessary to split these data further ahead into training, test, and validation sets.

## Model Architecture

The machine learning model employed in this project utilizes a convolutional neural network (CNN) architecture. CNNs are well-suited for image classification tasks, as they can automatically learn hierarchical representations of features directly from pixel values. The model comprises multiple convolutional layers followed by max-pooling layers to extract and downsample features, followed by fully connected layers for classification. Dropout and batch normalization layers are incorporated to prevent overfitting and improve model generalization.

## Approach

1. **Data Preprocessing**: The dataset will be preprocessed to enhance image quality, normalize features, and prepare it for model training.
2. **Model Development**: Machine learning and deep learning algorithms will be explored, including convolutional neural networks (CNNs), to build an accurate prediction model.
3. **Model Evaluation**: The developed models will be evaluated using performance metrics to assess their effectiveness in predicting malaria disease. 
The trained model is evaluated using the testing dataset to assess its generalization performance on unseen data.
Evaluation metrics such as accuracy are computed to quantify the model's predictive performance and assess its effectiveness in malaria diagnosis.
5. **Deployment**: Once trained and evaluated, the model can be deployed in real-world applications to assist healthcare professionals in diagnosing malaria more accurately and efficiently. The model can be integrated into existing healthcare systems or deployed as a standalone application accessible through web or mobile platforms. Continuous monitoring and updates may be required to ensure the model's performance remains robust and up-to-date with emerging data and clinical insights. The final model will be deployed as a web application or API for easy access and utilization by healthcare professionals.

# Utility of the models created

-Improved Diagnosis Accuracy
-Early Detection and Treatment
-Support for Remote Healthcare

## Future Improvements

Future iterations of the project may include:
- Increase sub sample size to refine models and avoid overfitting
  
To ensure the capacity of expandability, as the models can be continually refined and adapted based on feedback, new data, and advancements in machine learning techniques, We intend to develop automated training and deployment pipelines that can be easily configured and adapted for different models and regularization techniques. This facilitates experimentation with different model configurations and enables rapid iteration and deployment of new versions.

## Contributors

- Liliana Alvelos
- Carlos Carneiro
