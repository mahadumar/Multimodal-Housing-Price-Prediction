# Multimodal Housing Price Prediction

## Objective
Build a machine learning model that predicts housing prices using both structured tabular data and house images.

## Methodology
1. **Data Processing**: 
   - Tabular data: Handled numerical and categorical features with scaling and encoding
   - Image data: Preprocessed using VGG16-specific preprocessing

2. **Model Architecture**:
   - Image branch: Pretrained VGG16 with custom top layers
   - Tabular branch: Dense neural network
   - Fusion: Concatenation of both modalities followed by regression layers

3. **Training**:
   - 50 epochs with early stopping potential
   - Adam optimizer with MSE loss
   - Batch size of 32

4. **Evaluation**:
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Error distribution analysis

## Key Results
- MAE: $234,277.00
- RMSE: $267,265.59
- The multimodal approach effectively combines different data types
- Model performance varies across different price ranges

## Files
- `Multimodal ML.ipynb`: Complete Jupyter notebook
- `housing_price_multimodal_model.h5`: Trained model
- `tabular_preprocessor.pkl`: Preprocessing pipeline for tabular data

## Usage
1. Load the model: `model = keras.models.load_model('housing_price_multimodal_model.h5')`
2. Load the preprocessor: `preprocessor = joblib.load('tabular_preprocessor.pkl')`
3. Preprocess new data: `tabular_processed = preprocessor.transform(new_tabular_data)`
4. Preprocess images: Use the `load_and_preprocess_image` function
5. Make predictions: `predictions = model.predict([images, tabular_processed])`

## Future Improvements
1. Use real housing data with actual images
2. Implement more advanced fusion techniques (attention mechanisms)
3. Add more sophisticated image augmentation
4. Experiment with different CNN architectures (ResNet, EfficientNet)
5. Implement hyperparameter tuning
6. Add uncertainty estimation to predictions
