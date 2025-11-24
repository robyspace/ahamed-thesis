#!/usr/bin/env python3
"""
Re-save the model with pickle protocol 4 for better compatibility
"""
import joblib
import sys

# Load the existing model
print("Loading model...")
model = joblib.load('variance_model_best.pkl')

# Re-save with protocol 4
print("Re-saving model with protocol 4...")
joblib.dump(model, 'variance_model_best.pkl', protocol=4)

print("Model re-saved successfully!")
