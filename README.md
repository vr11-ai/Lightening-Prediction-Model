# Lightening-Prediction-Model
This contains the models for predicting lightening events mainly in prayagraj region.

Spatio-Temporal Lightning Prediction Pipeline
A highly reproducible machine learning framework developed in lightning_prediction.py to predict regional lightning strikes. Using localized meteorological data (such as the Prayagraj dataset), this pipeline addresses extreme class imbalance using a unique hybrid data augmentation approach and benchmarks four optimized machine learning algorithms.  

🚀 Key Framework FeaturesTemporal Feature Engineering: Extracts underlying diurnal and seasonal patterns by deriving hour and month parameters directly from timestamps.  Robust Imbalance Mitigation: Combines minority class oversampling (generating 500 synthetic lightning samples using a 1% Gaussian noise injection) with controlled majority class downsampling to force the models to learn clear decision boundaries.  Meteorological Validation Metrics: Evaluates models using Balanced Accuracy and Probability of Detection (POD / Recall). POD is safety-critical for early warning networks to minimize missed hazardous events.  Strict Reproducibility: Features a hardcoded global seed architecture (RANDOM_SEED = 42) across the data splits, synthetic noise generation, and model parameters to ensure identical run-to-run execution.  

🛠️ Tech Stack & DependenciesData Manipulation: pandas, numpy  Machine Learning Framework: scikit-learn  Gradient Boosting Ecosystem: xgboost  Environment: Formatted for deployment on Google Colab or local Python environments  

🧠 Benchmarked ModelsThe pipeline in lightning_prediction.py evaluates and logs comparative results for four powerful algorithmic families, each constrained to prevent overfitting:  
K-Nearest Neighbors (KNN): Tuned with k=11 for smoother spatial/proximity decision boundaries.  
Gradient Boosting Classifier (GBoost): Constrained with a shallow max_depth=3 and 50 estimators to prevent variance spikes.  
Support Vector Classifier (SVM): Implemented with an RBF kernel and a regularization penalty of C=0.5.  
XGBoost: Configured with robust L1 (reg_alpha=0.1) and L2 (reg_lambda=1) regularization rules.  

📄 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.
