# Nerual Network Charity Analysis

## Overview of the Analysis: 
This analysis aimed to assess effectiveness and positive impact of foundation dollars distributed to different charities and to create a sophisticated analysis model which could accurately identify charities worthy of continued financial support. The data set had a limited number of features for just over 34,000 organizations. Instead of using statistical and machine learnning models, this analysis was attempted using a deep-learning neural network. 

## Results: 
### Data Preprocessing
The dataset was preprocessed by identifying the value of variables, dropping uneccessary columns, binning when possible, and encoding bins to binary values. 
  - Target variable: `IS_SUCCESSFUL`
  - Feature variables: `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT`
  - Removed variables: `EIN`, `NAME`
  
### Compiling, Training, and Evaluating the Model
Using `IS_SUCCESSFUL` as the feature value, the dataset was split into training and testing data and fitted with a standard scaler. The original model was defined as follows:
 
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
    
And resulted in only a 72% accuracy level:  

    268/268 - 0s - loss: 0.5570 - accuracy: 0.7214 - 270ms/epoch - 1ms/step
    Loss: 0.5569565892219543, Accuracy: 0.7213994264602661

Many attempts were made to achieve a higher accuracy percentage, with the modest goal of at least 75%.  This was attempted by increasing the number of 

       


## Summary: 
A goal of at least 75% accuracy could not be met using the planned analysis model despite avarious attempts applying variety of adjustments to the model. 
