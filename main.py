import gradio as gr
import pandas as pd
import joblib


def predict_diagnosis(radius_worst_param, area_worst_param, perimeter_worst_param,
                      concave_points_worst_param, concave_points_mean_param):
    """
    Predict malignant or benign tumour.

    :param radius_worst_param: "worst" or largest mean radius
    :param area_worst_param: "worst" or largest mean area
    :param perimeter_worst_param: "worst" or largest mean perimeter
    :param concave_points_worst_param: "worst" or largest mean number of concave portions of the contour
    :param concave_points_mean_param: mean number of concave portions of the contour
    :return: malignant or benign diagnosis
    """
    # Load SVM model
    svm_model = joblib.load('saved_objects/reduced_features_svc.pkl')
    # Load scaler
    scaler = joblib.load('saved_objects/reduced_features_scaler.pkl')

    # Model Features
    features = ['perimeter_worst', 'concave points_worst', 'area_worst', 'concave points_mean', 'radius_worst']
    # Values from GUI
    form_values = [[perimeter_worst_param, concave_points_worst_param, area_worst_param,
                    concave_points_mean_param, radius_worst_param]]

    # Convert values to DataFrame
    form_values = pd.DataFrame(form_values, columns=features)
    # Scale values
    form_values_scaled = scaler.transform(form_values)

    # Predict
    prediction = svm_model.predict(form_values_scaled)

    if prediction == 'M':
        diagnosis = 'Malignant Tumour'
    else:
        diagnosis = 'Benign Tumour'

    return diagnosis


if __name__ == "__main__":

    # User Input
    radius_worst = gr.Slider(label='Enter the largest mean radius',
                             minimum=0, maximum=50, step=0.001)
    area_worst = gr.Slider(label='Enter the largest mean area',
                           minimum=0, maximum=4500, step=0.1)
    perimeter_worst = gr.Slider(label='Enter the largest mean perimeter',
                                minimum=0, maximum=300, step=0.01)
    concave_points_worst = gr.Slider(label='Enter the largest mean number of concave portions of the contour',
                                     minimum=0, maximum=0.5, step=0.000001)
    concave_points_mean = gr.Slider(label='Enter the mean number of concave portions of the contour',
                                    minimum=0, maximum=0.25, step=0.000001)

    # Program Output
    output = gr.Textbox()

    # Gradio Application
    app = gr.Interface(fn=predict_diagnosis,
                       inputs=[radius_worst, area_worst, perimeter_worst, concave_points_worst, concave_points_mean],
                       outputs=output)
    app.launch()
