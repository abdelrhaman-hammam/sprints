Developed as part of the Sprints Program, this project is a machine learning pipeline for heart disease prediction. Utilizing clinical data from the UCI Machine Learning Repository (Heart Disease Dataset, featuring 13 clinical features such as age, blood pressure, and cholesterol levels; target: presence of heart disease), the application predicts heart disease likelihood with high accuracy, offering a valuable decision-support tool for healthcare professionals. The system encompasses data processing and cleaning, feature selection and dimensionality reduction, training of multiple machine learning models with hyperparameter optimization, and deployment of a user-friendly Streamlit web application for real-time risk predictions. This end-to-end solution was built following industry best practices and production-level standards during an intensive training program.


Heart_Disease_Project/
│
├── data/
│   ├── heart_disease_cleaned.csv          # Cleaned and processed data
│   ├── heart_disease_selected.csv         # Feature-selected data
│   └── heart_disease_pca.csv              # PCA-transformed data
│
├── models/
│   ├── final_model.pkl                    # Trained model
│   ├── scaler.pkl                         # Feature scaler
│   ├── pca_model.pkl                      # PCA transformation model
│   └── deployment/
│       └── model_package.pkl              # Complete deployment package
│
├── notebooks/
│   ├── 01_data_preprocessing.py           # Data cleaning and preparation
│   ├── 02_pca_analysis.py                 # Dimensionality reduction
│   ├── 03_feature_selection.py            # Feature selection methods
│   ├── 04_supervised_learning.py          # Model training
│   ├── 05_unsupervised_learning.py        # Clustering analysis
│   └── 06_hyperparameter_tuning.py        # Model optimization
│
├── results/
│   ├── evaluation_metrics.csv             # Performance metrics
│   ├── feature_importance.csv             # Feature importance scores
│   └── visualizations/                    # Analysis plots and charts
│
├── app.py                                 # Streamlit web application
├── requirements.txt                       # Python dependencies
├── run_pipeline.py                        # Automated pipeline runner
└── README.md                              # Project documentation
