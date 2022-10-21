# Neural Network Charity Analysis
## Overview of the Analysis
### Purpose 
Alphabet Soup, a non-profit foundation, desires to analyze the impact of their donations and wants to vet potential recipients by predicting which organizations are worth donating to and which are too high risk. A deep learning neural network model was designed using the Python TensorFlow library to evaluate all types of input data and produce a clear decision making result.
## Results 
### Data Preprocessing 
* The IS_SUCCESSFUL column is considered the target for the model.
* The APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, and ASK_AMT columns are considered to be the features of the model.
* The EIN, NAME, STATUS, and SPECIAL_CONSIDERATION columns are neither targets nor features and should be removed from the input data.
![process 1](https://user-images.githubusercontent.com/106560739/197108407-28d8e1f5-e0ad-43a6-bdf0-cf118798d892.png)
![process 2](https://user-images.githubusercontent.com/106560739/197108411-de074b80-5957-40c2-918b-b0a913ccfe27.png)
![process 3](https://user-images.githubusercontent.com/106560739/197108417-0968ba9f-9f38-4720-ba59-a0c4bf35646b.png)
### Compiling, Training, and Evaluating the Model
* 120 neurons were selected with a sigmoid function for the first layer, 50 nuerons with a ReLU function for the second, and 18 neurons with for the third, and a sigmoid function for the outer layer. The activation function was changed for the first layer because it increased the model's performance.
* This achieved an accuracy of 69% and was not able to achieve the target model performance.
* In an attempt to increase the model performance the following was enacted: dropping more columns, creating more bins for rare occurances in columns, decreasing the number of values in some bins, adding more neurons to the hidden layers, using a differnet activation function, and increasing the number of epochs.
![model](https://user-images.githubusercontent.com/106560739/197108662-2d549c3b-1237-4028-ab92-31f30ae43867.png)
![model output](https://user-images.githubusercontent.com/106560739/197108668-181fb266-0c8a-4863-a474-2c82d9682d31.png)
![model compile and callback](https://user-images.githubusercontent.com/106560739/197108712-18d85075-c253-4e15-b1f5-81f96c6a7b0e.png)
![model train](https://user-images.githubusercontent.com/106560739/197108718-cd13cc7c-db67-4e20-b1b2-e78f85bdefca.png)
![model evaluate](https://user-images.githubusercontent.com/106560739/197108724-179818b9-4a9c-4bd0-99c1-b2eb9955862f.png)
## Summary
Through the removal of noisy features, additional neurons and hidden layers and changed activation functions, the accuracy of the optimized model for predicting whether a donation is successful ended up being 0.69703 and its loss metric was 1.1497.
### Recommendation
This classification problem could be solved by using a random forest model since it would randomly sample the preprocessed data and build several smaller, less complex decision trees. A benefit of using a random forest model is how robust it is against overfitting of the data because all of the weak learners are trained on different pieces of the data. Moreover, it can be used to effectively rank the importance of input variables and it is robust to outliers and nonlinear data. Finally, it can run efficiently on large datasets.
