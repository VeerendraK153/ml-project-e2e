from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)
pipeline = PredictPipeline()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get all form data
        data = {key: request.form[key] for key in request.form.keys()}

        # Explicitly convert numeric fields
        numeric_cols = [
            'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
            'GrLivArea', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'YearBuilt', 'YearRemodAdd'
        ]

        for col in numeric_cols:
            if col in data:
                try:
                    data[col] = float(data[col]) if data[col] else 0.0
                except ValueError:
                    data[col] = 0.0

        price = pipeline.predict(data)
        return render_template('index.html', prediction_text=f'Predicted House Price: ${price:.2f}')
    else:
        return render_template('index.html', prediction_text=None)

if __name__ == '__main__':
    app.run(debug=True)
