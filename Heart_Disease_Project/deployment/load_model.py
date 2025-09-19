import joblib
import pandas as pd
import numpy as np

class HeartDiseasePredictor:
    def __init__(self, model_path):
        """Load the deployed model package"""
        self.deployment_package = joblib.load(model_path)
        self.model = self.deployment_package['model']
        self.feature_names = self.deployment_package['feature_names']
        self.model_info = self.deployment_package
        
    def predict(self, input_data):
        """Make prediction on new data"""
        
        if isinstance(input_data, (list, np.ndarray)):
            input_df = pd.DataFrame(input_data, columns=self.feature_names)
        else:
            input_df = input_data[self.feature_names]
            
        return self.model.predict(input_df)
    
    def predict_proba(self, input_data):
        """Get prediction probabilities"""
        if isinstance(input_data, (list, np.ndarray)):
            input_df = pd.DataFrame(input_data, columns=self.feature_names)
        else:
            input_df = input_data[self.feature_names]
            
        return self.model.predict_proba(input_df)
    
    def get_model_info(self):
        """Get model information"""
        return self.model_info
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None


if __name__ == "__main__":
    
    predictor = HeartDiseasePredictor(
        "C:/studies/sprints/Heart_Disease_Project/deployment/model_package.pkl"
    )
    
    print("Model loaded successfully!")
    print("Model type:", predictor.model_info['model_type'])
    print("Accuracy:", predictor.model_info['best_accuracy'])
    print("Features:", predictor.feature_names)