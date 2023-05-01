# Deep-Learning-Challenge Report
Week 21 Module

**Background** <br>

The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

-------------------------------------------


**Overview** <br>

A deep learning model can identify and account for more information than any number of neurons in any single hidden layer. However, it will require longer training iterations and memory resources than their basic neural network counterparts to achieve higher degrees of accuracy and precision.

Step 1: Preprocess the Data
  - Use charity_data.csv containing more than 34,000 organisations that have received funding from Alphabet Soup over the years.
  - Cleaning data by dropping the `EIN` and `NAME` columns.
  - Make `IS_SUCCESSFUL` as y value (target) for the model
  - Make the rest of clean dataframe without the y value as X value (features) for the model.
  - Determine the number of unique values for each column.
  - Determine the number of data points for each unique value.
  - Choose a cutoff value and create a list of classifications (<600) and applications (<700) and bin rare categorical values together into a new value called "Other".
  - Convert categorical data to numeric with `pd.get_dummies`
  - Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
  - Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the transform function.

Step 2: Compile, Train, and Evaluate the Model
  - Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
  - Create the first hidden layer and choose an appropriate activation function.
  - Create an output layer with an appropriate activation function.
  - Check the structure of the model.
  - Compile and train the model.
  - Create a callback that saves the model's weights every five epochs.
  - Evaluate the model using the test data to determine the loss and accuracy.

Step 3: Optimise the Model
  - Adjust the input data to ensure that no variables or outliers are causing confusion in the model.
  - Add more neurons to a hidden layer.
  - Add more hidden layers.
  - Use different activation functions for the hidden layers.
  - Add or reduce the number of epochs to the training regimen.

-------------------------------------------

**Results**

Attempt 1

  - First Layer: 15 units;  Activation = ReLu
  - Second Layer: 30 units;  Activation = ReLu
  - Epoch: 50
  - Output Layer = Sigmoid

![Model1](https://user-images.githubusercontent.com/117326039/235396478-4e12475f-1834-4241-a626-521e4b38bab1.png)

  - For this first attempt, it resulted to 72.60% Accuracy.
  - It can still be improved for higher precision and degree of accuracy.
![Model_1](https://user-images.githubusercontent.com/117326039/235396248-fa7c023d-74fa-4ce5-847e-05aee4abf85b.png)



Attempt 2 

  -Increase neurons, add aditional layers and use tanh activation for the new layers.
  -First Layer: 30 units; Activation = Relu
  -Second Layer: 60 units; Activation = Relu
  -Third Layer: 120 units; Activation = tanh
  -Fourth Layer: 180 units; Activation = tanh
  
![Model2](https://user-images.githubusercontent.com/117326039/235396615-c380969e-264b-4914-911c-710bfaef7be6.png)

 -For this second attempt, it resulted to 72.54% Accuracy.

![Model_2](https://user-images.githubusercontent.com/117326039/235396630-ec5aad6f-5562-4642-a8d1-b05ff929f3ff.png)



Attempt 3

  -Increase neurons, add another layer and reduce epoch.
  -First Layer: 50 units; Activation = Relu
  -Second Layer: 100 units; Activation = Relu
  -Third Layer: 150 units; Activation = tanh
  -Fourth Layer: 200 units; Activation = tanh
  -Fifth Later: 250 units; Activation = sigmoid
  -Sixth Later: 300 units; Activation = sigmoid
  -Epoch: 20
  
  

![Model3](https://user-images.githubusercontent.com/117326039/235401497-9cafe0f9-722b-4fff-b104-1a566ddfd102.png)


  - Final attempt resulted to the highest accuracy for this model, 72.73%
![Model_3](https://user-images.githubusercontent.com/117326039/235397933-e0141b1f-c035-4d0b-b0ba-4b8fa90d5b8c.png)


------------------------------------------

**Summary**

For this deep learning model analysis, the targeted accuracy of 75% were not achieved on the three attempts. By experimenting with increasing the neurons, adding hidden layers, reducing epoch and changing the activation achieved my highest accuracy which is 72.73%. 

If I would do another attempt, I would use the Keras sequential optimisation model to automatically hypertune the parameters to get highest accuracy possible.




