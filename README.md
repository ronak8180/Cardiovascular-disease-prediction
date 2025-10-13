# Heart Disease Prediction System

A web application that predicts heart disease risk using machine learning.

## Features

- **Multi-page Interface**: Dashboard, Prediction Form, History, and About pages
- **Interactive Forms**: User-friendly input fields with real-time validation
- **AI Prediction**: Random Forest model with 84% accuracy
- **Visual Analytics**: Interactive charts and risk analysis
- **Responsive Design**: Works on desktop and mobile devices

## Technology Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **ML Model**: Random Forest Classifier (Scikit-learn)
- **Data Processing**: Pandas, NumPy
- **Data Balancing**: SMOTE (imbalanced-learn)

## Local Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**:
   ```bash
   python app.py
   ```

3. **Access the Application**:
   - Open your browser and go to `http://localhost:5000`

## Deployment

### Heroku Deployment

1. **Create Heroku Account**: Sign up at [heroku.com](https://heroku.com)

2. **Install Heroku CLI**: Download from [devcenter.heroku.com](https://devcenter.heroku.com/articles/heroku-cli)

3. **Login to Heroku**:
   ```bash
   heroku login
   ```

4. **Create Heroku App**:
   ```bash
   heroku create your-app-name
   ```

5. **Deploy**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push heroku main
   ```

6. **Open Your App**:
   ```bash
   heroku open
   ```

### Railway Deployment

1. **Create Railway Account**: Sign up at [railway.app](https://railway.app)

2. **Connect GitHub**: Link your repository

3. **Deploy**: Railway will automatically detect and deploy your Flask app

## Model Information

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 84%
- **ROC AUC Score**: 0.92
- **Features**: 21 health and lifestyle factors
- **Data Source**: CDC Behavioral Risk Factor Surveillance System

## API Endpoints

- `GET /` - Redirects to dashboard
- `GET /dashboard` - Main dashboard page
- `GET /predict` - Prediction form page
- `POST /predict` - Submit prediction request
- `GET /history` - Prediction history page
- `GET /about` - About and help page

## File Structure

```
├── app.py                 # Main Flask application
├── processed_data.csv     # Training dataset
├── templates/             # HTML templates
│   ├── index.html        # Prediction form
│   ├── dashboard.html    # Dashboard page
│   ├── history.html      # History page
│   └── about.html        # About page
├── static/               # Static files
│   └── style.css         # CSS styles
├── requirements.txt      # Python dependencies
├── Procfile             # Heroku deployment file
├── runtime.txt          # Python version specification
└── README.md            # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.

## Disclaimer

This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical concerns.