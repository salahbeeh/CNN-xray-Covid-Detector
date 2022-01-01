
# Computational Biology  
# Capstone Project
## Project: Pneumonia Classification Using Deep Learning from Chest X‑ray Images During COVID‑19


## Inspiration / References
Ibrahim, A. U., Ozsoz, M., Serte, S., Al-Turjman, F., & Yakoi, P. S. (2021). _Pneumonia Classification Using Deep Learning from Chest X-ray Images During COVID-19. Cognitive Computation._ doi:10.1007/s12559-020-09787-5
(https://pubmed.ncbi.nlm.nih.gov/33425044/)

## Project Overview
Convolutional neural network (CNN), a leading DL tool that is popularly used in different fields of healthcare system due to their ability to extract features and learn to distinguish between different classes (i.e., positive and negative, infected and healthy, cancer and non-cancer, etc.). our model can detect Covid-19 x-rays form normal ones.
  
Keywords: Deep learning, ANN, Feature Extraction, CNN, Covid-19, Machine Recognition, 

## Problem Statement
Back in 2019, when Covid-19 first hit Wuhan, China. doctors got confused about the new disease based on the common symptoms between it and pneumonia and they took different approaches to diagnose pneumonia, some of these approaches include chest X-rays,
CT scans (which form the basis of our contribution) and other approaches like sputum test, pulse oximetry, thoracentesis, blood gas analysis, bronchoscopy, pleural fluid culture, complete blood count, etc. And still it was difficult to be certain if that was Covid or not.

## Domain Background
COVID-19 is an especially contagion caused by severe acute metastasis syndrome coronavirus 2 (SARCoV-2) that is that the recent disease that's caused by one amongst
the members of the family of Coronaviridae family.  The first case of COVID-19 was rumored in Wuhan, and reported in  31st December 2019. 

The virus unfold from town to city and from one country to a different resulting in a global health crisis. However, it had been not till March 11, 2020 that WHO declared it as pandemic.

COVID-19 can be transmitted through metastasis droplets that are exhaled or secreted by infected persons. Coronaviruses invade the lung’s alveoli (an organ answerable for exchange of O2 and CO2), thereby inflicting pneumonia. The symptoms of COVID-19 embrace dry cough, fatigue, fever, septic shock, organ failure, anorexia, dyspnea, myalgias, phlegm secretion severe pneumonia, acute respiratory distress syndrome (ARDS), etcetera The pandemic caused by SAR-CoV-2 is consider deadly even after WHO has approved some  vaccines.

![metrics](https://miro.medium.com/max/1000/1*fxiTNIgOyvAombPJx5KGeA.png)

My personal **motivation** is to contribute some ideas to help detect such virus especially using x-rays and CL which is available and less costly compared to blood tests 

The link to my data source is:

(https://github.com/ieee8023/covid-chestxray-dataset)
(https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

### Requirement
### Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](http://matplotlib.org/)
- [scikit-learn](http://scikit-learn.org/stable/)
- [seaborn](https://seaborn.pydata.org/)
- [tensorflow](https://www.tensorflow.org/)
- [keras](https://keras.io/)


You will also need to have software installed to run and execute a [Jupyter Notebook](http://ipython.org/notebook.html)

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. 

### Code

code is provided in the `Covid.ipynb` notebook file. 

### Run

In a terminal or command window, navigate to the top-level project directory `cnnCovidDetector/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Covid.ipynb
```  
or
```bash
jupyter notebook Covid.ipynb
```

This will open the Jupyter Notebook software and project file in your browser.

### Data
The data collected form the datasets above contains 196 PA x-rays of Covid positive cases collected from the following GitHub repo ( https://github.com/ieee8023/covid-chestxray-dataset), and another 196 PA x-rays of Covid negative cases that shares the symptoms of Covid for the following Kaggle (https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

### Refinement
Considering the above paper that used an **Alexnet** model, my benchmark model showed that the results were promising with a 96% classification accuracy rate on testing images. I plan to work on improving the performance of detecting the positive covid x-rays form normal one.

but Increasing the number of **CNN** layers and use convolutional, using **max pooling** ,**flatten**,  **relu and sigmoid activation function** and Optimizing the CNN using **Adam**, all together did improve the accuracy.

## Solution Statement
A Convolutional Neural Network (CNN) is a special type of feed-forward multilayer trained in supervised mode. The CNN trained and tested our database that contains 392 of chest positive and negative corona x-rays . 

## Model Evaluation and Validation 

The final architecture is chosen because they performed the best results among the tried combination (the architecture in jupyter notebook) The complete description of the final model: - 
* The kernel size of the first two convolution layers is 3 * 3. 
* The first two convolutional layers. 
  - one learned by 32 filters followed by 64 filters and they use a **relu** activation function. 
  - The max pool layer has a pool size of 2 * 2. 
  - The dropout rate is 0.25. 
 * The kernel size of the third and the fourth convolution layers is 3 * 3. 
   - The third and the fourth convolution layers learned 64 filters followed by 128 filters, using a relu activation function. 
   - The max pool layer has a pool size of 2 * 2.
   - The dropout rate is 0.3. 
   - The first dense layers have 64 units and use a relu activation function. 
   - The last dense layers have28 units and use a sigmoid activation function. 

This model can detect positive Covid x-rays from normal ones 
- The model generalizes well to unseen data it’s predicted the label perfectly. 
- The model didn’t affect with small changes in the data, or to outliers because of scaling of the data between values 0 to 1. 
- We can trust in the model because of it us a high accuracy after fitting the Neural Network.

## Evaluation Metrics
Generating a confusion matrix, for summarizing the performance of a classification algorithm. Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset. Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.


