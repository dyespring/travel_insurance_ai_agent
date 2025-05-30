import csv
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    # Read the CSV file and extract unique destinations
    destinations = set()  # Using a set to automatically handle uniqueness
    
    with open('travel-insurance.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            destinations.add(row['Destination'])
    
    # Convert to sorted list for consistent ordering
    sorted_destinations = sorted(destinations)
    
    return render_template('index.html', options=sorted_destinations)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = request.form.get('age')
        gender = request.form.get('gender')
        duration = request.form.get('duration')
        destination = request.form.get('destination')
        
        # Here you would typically:
        # 1. Process the form data
        # 2. Generate predictions
        # 3. Prepare results for the template
        
        # For now, we'll just pass the form data to the template
        return render_template('predict.html', 
                             age=age,
                             gender=gender,
                             duration=duration,
                             destination=destination)
    
    # If it's a GET request, redirect to home
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)