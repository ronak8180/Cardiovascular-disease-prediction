import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import os

# Generate a sample feature importance chart
feature_names = ['Age', 'BMI', 'Exercise', 'Smoking', 'Diabetes', 'Diet']
importances = [25, 20, 15, 18, 12, 10]
plt.figure(figsize=(6, 4))
plt.bar(feature_names, importances, color='#667eea')
plt.title('Feature Importance')
plt.ylabel('Importance (%)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Generate a sample prediction result chart
labels = ['No Heart Disease', 'Heart Disease Risk']
values = [60, 40]
plt.figure(figsize=(4, 4))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['#27ae60', '#e74c3c'])
plt.title('Sample Prediction Result')
plt.tight_layout()
plt.savefig('prediction_result.png')
plt.close()

# Create PDF report
pdf_path = 'Cardiovascular_Disease_Prediction_Report.pdf'
c = canvas.Canvas(pdf_path, pagesize=A4)
width, height = A4

# Title
c.setFont('Helvetica-Bold', 22)
c.drawString(50, height - 60, 'Cardiovascular Disease Prediction Model Report')

# Project Overview
c.setFont('Helvetica', 14)
c.drawString(50, height - 100, 'Project Overview:')
c.setFont('Helvetica', 12)
c.drawString(50, height - 120, 'This project predicts the risk of cardiovascular disease using health data.')
c.drawString(50, height - 135, 'It uses a machine learning model deployed with a Flask web app on AWS.')

# Data Flow Diagram (placeholder)
c.setFont('Helvetica-Bold', 14)
c.drawString(50, height - 170, 'Data Flow Diagram:')
c.setFont('Helvetica', 12)
c.drawString(50, height - 190, 'User Input → Data Preprocessing → Model Prediction → Results & Visualization')
c.setStrokeColor(colors.blue)
c.line(50, height - 200, 400, height - 200)

# Model Architecture (placeholder)
c.setFont('Helvetica-Bold', 14)
c.drawString(50, height - 230, 'Model Architecture:')
c.setFont('Helvetica', 12)
c.drawString(50, height - 250, 'Random Forest Classifier with feature engineering and scaling.')
c.drawString(50, height - 265, 'Trained on processed health data (CSV).')

# Feature Importance Chart
c.setFont('Helvetica-Bold', 14)
c.drawString(50, height - 300, 'Feature Importance:')
if os.path.exists('feature_importance.png'):
    c.drawImage(ImageReader('feature_importance.png'), 50, height - 500, width=300, height=150)

# Prediction Result Chart
c.setFont('Helvetica-Bold', 14)
c.drawString(50, height - 520, 'Sample Prediction Result:')
if os.path.exists('prediction_result.png'):
    c.drawImage(ImageReader('prediction_result.png'), 50, height - 700, width=200, height=150)

# Deployment Details
c.setFont('Helvetica-Bold', 14)
c.drawString(50, height - 730, 'Deployment & Usage:')
c.setFont('Helvetica', 12)
c.drawString(50, height - 750, '• Deployed on AWS Elastic Beanstalk')
c.drawString(50, height - 765, '• Source code on GitHub: github.com/ronak8180/Cardiovascular-disease-prediction')
c.drawString(50, height - 780, '• Export features: CSV and PDF downloads for predictions')

# Recommendations
c.setFont('Helvetica-Bold', 14)
c.drawString(50, height - 810, 'Recommendations & Future Improvements:')
c.setFont('Helvetica', 12)
c.drawString(50, height - 830, '• Add more health features and improve model accuracy')
c.drawString(50, height - 845, '• Integrate real-time data and notifications')
c.drawString(50, height - 860, '• Enhance UI with more interactive charts')

c.save()

print(f'Report generated: {pdf_path}')
