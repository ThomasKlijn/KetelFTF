# Ketel & FTF Processing Application

## Overview
This is a machine learning-powered FastAPI application that processes Excel files to predict and classify boiler-related maintenance work orders. The application uses trained models to classify work orders as boiler-related ("Ketel gerelateerd") and first-time-fixed ("FTF") status.

## Project Architecture
- **Backend**: FastAPI application serving machine learning predictions
- **Frontend**: Simple HTML interface for file upload and download
- **Models**: Pre-trained scikit-learn models with TF-IDF vectorizers
- **File Processing**: Excel file processing with color-coded predictions

## Recent Changes
- **2025-09-11**: Successfully imported from GitHub and configured for Replit environment
  - Installed all Python dependencies (FastAPI, pandas, numpy, scikit-learn, joblib, openpyxl)
  - Configured FastAPI workflow on port 5000
  - Verified machine learning models are accessible
  - Set up deployment configuration for autoscale

## Key Features
1. **Excel File Processing**: Upload Excel files for automated classification
2. **Machine Learning Predictions**: Uses keyword matching and ML models for classification
3. **Output Splitting**: Separate files based on classification results
4. **Color Coding**: Visual indicators for prediction confidence levels

## Machine Learning Models
- `ketel_model_Vfinal.joblib`: Model for classifying boiler-related work orders
- `ketel_vectorizer_Vfinal.joblib`: TF-IDF vectorizer for boiler classification
- `ftf_model_Vfinal.joblib`: Model for first-time-fixed predictions
- `ftf_vectorizer_Vfinal.joblib`: TF-IDF vectorizer for FTF classification

## API Endpoints
- `GET /`: Serves the main HTML interface
- `POST /process_excel/`: Processes uploaded Excel files and returns predictions
- `POST /split_output/`: Splits processed files based on classification results

## Technical Stack
- **Python 3.11**
- **FastAPI**: Web framework
- **pandas**: Data processing
- **scikit-learn**: Machine learning
- **openpyxl**: Excel file handling
- **uvicorn**: ASGI server

## Configuration
- **Port**: 5000 (configured for Replit environment)
- **Host**: 0.0.0.0 (allows external access)
- **Deployment**: Autoscale (stateless web application)