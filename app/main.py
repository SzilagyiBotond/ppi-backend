from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import pandas as pd
import io
import zipfile
from pathlib import Path
import uuid

from .data_loading import load_dataframe
from .visualization import generate_all_patient_plots
from .model import prepare_data, train_and_evaluate_models

app = FastAPI()

@app.post("/get-cross-correlation/")
async def generate_cross_correlation_plots(file: UploadFile = File(...)):
    """
    Process uploaded Excel file and return correlation plots as a zip file
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Process data
        processed_df = load_dataframe(df)
        
        # Generate correlation plots
        all_plots = generate_all_patient_plots(processed_df)
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add each plot to the zip
            for plot_name, plot_bytes in all_plots.items():
                zip_file.writestr(f"{plot_name}.png", plot_bytes)
        
        # Prepare the response
        zip_buffer.seek(0)
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=correlation_plots.zip"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-data/")
async def generate_predictions_and_importance_extraction(file: UploadFile = File(...)):
    """
    Process uploaded Excel file and return correlation plots as a zip file
    """
    try:
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
        
        # Process data
        df = load_dataframe(df)

        target_variables = ['MR area cm2', 'MR VC mm']

        all_plots = {}
        # Run analysis for each target variable
        for target in target_variables:
            # Regression analysis
            X_train, X_test, y_train, y_test, feature_names = prepare_data(df, target, classification=False)
            regression_plots = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, target, classification=False)
            all_plots.update({f"regression_{k}": v for k, v in regression_plots.items()})
            # Classification analysis
            X_train, X_test, y_train, y_test, feature_names = prepare_data(df, target, classification=True)
            classification_plots = train_and_evaluate_models(X_train, X_test, y_train, y_test, feature_names, target, classification=True)
            all_plots.update({f"classification_{k}": v for k, v in classification_plots.items()})
                
        
        # Create zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add each plot to the zip
            for plot_name, plot_bytes in all_plots.items():
                zip_file.writestr(f"{plot_name}.png", plot_bytes)
        
        # Prepare the response
        zip_buffer.seek(0)
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=analysis_and_predictions_plots.zip"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))