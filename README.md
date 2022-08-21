# Nerual Network Charity Analysis

## Overview of the Analysis: 
This analysis aimed to assess effectiveness and positive impact of foundation dollars distributed to different charities and to create a sophisticated analysis model which could accurately identify charities worthy of continued financial support. The data set had a limited number of features for just over 34,000 organizations. Instead of using statistical and machine learning models, this analysis was attempted using a deep-learning neural network. 

## Results: 
### Data Preprocessing
The dataset was preprocessed by identifying the value of variables, dropping unecessary columns, binning when possible, and encoding bins to binary values. 
  - Target variable: `IS_SUCCESSFUL`
  - Feature variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
  - Removed variables: `EIN`, `NAME`
  
### Compiling, Training, and Evaluating the Model
Using `IS_SUCCESSFUL` as the feature value, the dataset was split into training and testing data and fitted with a standard scaler. The original model, after many attempts at adjusting the number of neurons on each node layer, not wanting to overfit, was defined as follows:
 
    number_input_features = len(X_train_scaled[0])
    hidden_nodes_layer1 = 9
    hidden_nodes_layer2 = 5

    nn = tf.keras.models.Sequential()
      # First hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))
      # Second hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))
      # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
    
Which generated the shape:

    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 9)                 396       
     dense_1 (Dense)             (None, 5)                 50        
     dense_2 (Dense)             (None, 1)                 6         
    =================================================================
    Total params: 452
    Trainable params: 452
    Non-trainable params: 0
    
And resulted consistently in only a 72% accuracy level for the testing data:  

    268/268 - 0s - loss: 0.5552 - accuracy: 0.7231 - 250ms/epoch - 932us/step
    Loss: 0.555247962474823, Accuracy: 0.7231487035751343

Wanting to reach the modest goal of at least 75%, further attempts were made at adding neurons, hidden layers, and altering the activation functions of those layers, as well as more closely examining the dataset and attempting to determine if dropping any other non-important variables might lessen the margin of error.  

Optimization attempt #1: 2 hidden layers (16, 8 neurons; tanh, relu), 1 output sigmoid layer, 100 epochs

    268/268 - 0s - loss: 0.5560 - accuracy: 0.7257 - 247ms/epoch - 923us/step
    Loss: 0.5559937953948975, Accuracy: 0.7257142663002014 

Optimization attempt #2: 3 hidden layers (32, 16, 8 neurons; tanh, LeakyReLU, sigmoid), 1 output sigmoid layer, 250 epochs

    268/268 - 0s - loss: 0.5518 - accuracy: 0.7301 - 259ms/epoch - 968us/step
    Loss: 0.5518033504486084, Accuracy: 0.7301457524299622

Optimization attempt #3: Dropped `SPECIAL_CONSIDERATIONS` columns, 3 hidden layers (32, 16, 8 neurons; tanh, LeakyReLU, sigmoid), 1 output sigmoid layer, 250 epochs

    268/268 - 0s - loss: 0.5523 - accuracy: 0.7308 - 260ms/epoch - 970us/step
    Loss: 0.5523256063461304, Accuracy: 0.7308454513549805

As shown above, accuracy percentages do inch closer with each adjustment, but only in extremely small increments.

## Summary: 
A goal of at least 75% accuracy could not be met using this analysis model despite avarious attempts applying a wide variety of adjustments to the model. It is unclear how the feature variables are tied to the target variable using this model for analysis, but other methods - even linear regression methods that pit the dependent variable on the additional feature variables independently might have more insights plotted togehter in layers, or on their own.  
