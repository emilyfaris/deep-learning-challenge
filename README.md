# deep-learning-challenge

# Alphabet Soup Deep Learning Model Analysis

## Overview
The purpose of this analysis is to build a deep learning model to predict whether applicants for funding from Alphabet Soup will be successful based on the provided data. By preprocessing the data, constructing and optimizing a neural network, and evaluating its performance, I aim to achieve a model that can accurately predict successful applications, thereby aiding Alphabet Soup in their decision-making process.

## Results

### Data Preprocessing

- **Target Variable:**
  - `IS_SUCCESSFUL`: Indicates whether the application was successful.

- **Feature Variables:**
  - All other variables after preprocessing and encoding, such as `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, and `ASK_AMT`.

- **Removed Variables:**
  - `EIN` and `NAME`: Dropped as they are ID variables do not contribute to the prediction of the target variable.

### Compiling, Training, and Evaluating the Model

- **Original Model:**
  - **Neurons and Layers:**
    - Input Layer: 80 neurons with ReLU activation
    - Hidden Layer: 30 neurons with ReLU activation
    - Output Layer: 1 neuron with Sigmoid activation
  - **Performance:**
    - Loss: 1.2938
    - Accuracy: 0.7060

For the original model, an input layer with 43 input dimensions and 80 units was used to match the number of input features after preprocessing. A single hidden layer with 30 neurons was chosen as a starting point. Both the input and hidden layer used a ReLU activtion function to capture non-linear relationships in the data. The output layer used a single neuron with a Sigmoid activation function to produce a probability score for the binary classification task. While the initial accuracy was moderately high, the high loss indicated that there was room for improvement in the model’s performance.

- **Optimization Attempts:**

  1. **Change Hidden Layer Activation Function to Tanh:**
     - Hidden Layer Activation: Tanh
     - **Performance:**
       - Loss: 0.7048
       - Accuracy: 0.5361

For the first optimization attempt, the hidden layer activation function was changed to Tanh. The Tanh activation function outputs values between -1 and 1, which can be useful in cases where the data is normalized and can help in faster convergence during training. However, the performance deteriorated, suggesting that the Tanh activation function was less effective for this dataset compared to ReLU.

  2. **Add an Additional Hidden Layer:**
     - Second Hidden Layer: 20 neurons with ReLU activation
     - **Performance:**
       - Loss: 0.7370
       - Accuracy: 0.5319

For the second optimization attempt, a second additional hidden layer was added. Adding more layers can help the model learn more complex patterns by increasing its depth. However, the additional hidden layer did not improve the performance, indicating that simply increasing the model depth without other changes was not beneficial.

  3. **Hyperparameter Tuning Using Keras-Tuner:**
     - **Best Hyperparameters:**
       - Activation: Sigmoid
       - First Hidden Layer Units: 6
       - Number of Hidden Layers: 4
       - Units per Hidden Layer: 
         - Layer 1: 3 units
         - Layer 2: 5 units
         - Layer 3: 9 units
         - Layer 4: 9 units
         - Additional layers added during tuning: 
           - Layer 5: 1 unit
           - Layer 6: 3 units
     - **Performance:**
       - Loss: 0.5697
       - Accuracy: 0.7359

For the last optimization attempt, Keras-Tuner was used for a systematic search for the best combination of hyperparameters to enhance the model’s performance. Keras-Tuner determined the optimal activation functions for the input and hidden layers is Sigmoid with 4 hidden layeras and significantly fewer neurons per layer than the original model. This change may help in preventing overfitting by reducing model complexity. The hyperparameter tuning significantly improved the model’s accuracy and reduced the loss, achieving the best performance among all attempts.

## Summary

- **Overall Results:**
  The initial model achieved an accuracy of 0.7060 and a loss of 1.2938. Through optimization techniques, the performance was improved. Changing the hidden layer activation function to Tanh and adding an additional hidden layer did not yield better model accuracy. However, hyperparameter tuning using Keras-Tuner improved the model's performance, achieving an accuracy of 0.7359 and a loss of 0.5697. 

- **Recommendation for Future Models:**
  While the deep learning model provided almost satisfactory results, exploring other machine learning models could further improve performance. For instance, implementing ensemble methods like Random Forest or Gradient Boosting could potentially yield better predictive accuracy for this classification problem. These models can capture complex patterns in the data and might provide more robust and interpretable results. In addition, the `feature_importances_` attribute within the Random Forest Classifier library allows the user to determine which features contribute the most to the model's predictions. This information can be used to identify less important features that could potentially be dropped, which might simplify the model and improve its performance.

By following these recommendations, Alphabet Soup can enhance their predictive capabilities, ensuring better decision-making and resource allocation for funding applicants.
