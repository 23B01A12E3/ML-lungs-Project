import pickle
import numpy as np
import os
from django.shortcuts import render

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "optimized_model.pkl")  # your model file
model = pickle.load(open(model_path, "rb"))

def predict(request):
    result = ""
    if request.method == "POST":
        try:
            # Collect all 15 features with real names
            features = [
                int(request.POST.get("age", 0)),
                int(request.POST.get("gender", 0)),
                int(request.POST.get("smoking", 0)),
                int(request.POST.get("yellow_fingers", 0)),
                int(request.POST.get("anxiety", 0)),
                int(request.POST.get("peer_pressure", 0)),
                int(request.POST.get("chronic_disease", 0)),
                int(request.POST.get("fatigue", 0)),
                int(request.POST.get("allergy", 0)),
                int(request.POST.get("wheezing", 0)),
                int(request.POST.get("alcohol", 0)),
                int(request.POST.get("coughing", 0)),
                int(request.POST.get("shortness_breath", 0)),
                int(request.POST.get("swallowing", 0)),
                int(request.POST.get("chest_pain", 0)),
            ]

            data = np.array([features])
            pred = model.predict(data)
            result = "High Risk of Lung Cancer" if pred[0] == 1 else "Low Risk of Lung Cancer"

        except Exception as e:
            result = f"Error: {str(e)}"

    return render(request, "index.html", {"result": result})
