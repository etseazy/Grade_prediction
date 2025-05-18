from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("linear_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        reading_score = float(request.form["reading_score"])
        writing_score = float(request.form["writing_score"])
        prep_course = request.form["prep_course"]

        # Encode prep_course: 'completed' → 1, 'none' → 0
        prep_value = 1 if prep_course == "completed" else 0

        # Feature array in the correct order
        input_features = np.array([[reading_score, writing_score, prep_value]])
        
        prediction = model.predict(input_features)[0]

        # Generate a personalized message
        if prediction >= 90:
            message = "Excellent performance expected!"
        elif prediction >= 70:
            message = "Good job, keep it up!"
        elif prediction >= 50:
            message = "Needs improvement."
        else:
            message = "At risk. Consider extra support."

        return render_template("index.html", 
                               prediction=round(prediction, 2),
                               message=message)

    except Exception as e:
        return render_template("index.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
