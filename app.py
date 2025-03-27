from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib
import csv

app = Flask(__name__)

# Load model và scaler
with open('models/rf.pkl', 'rb') as f:
    model = pickle.load(f)
# with open('models/ann.pkl', 'rb') as f:
#     ann = pickle.load(f)
scaler = joblib.load('models/scaler.pkl')

train_columns = [
    "Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies",
    "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)",
    "STDs (number)", "STDs:condylomatosis", "STDs:vaginal condylomatosis",
    "STDs:vulvo-perineal condylomatosis", "STDs:syphilis", "STDs:pelvic inflammatory disease",
    "STDs:genital herpes", "STDs:molluscum contagiosum", "STDs:HIV", "STDs:Hepatitis B", "STDs:HPV",
    "STDs: Number of diagnosis", "Smokes_0.0", "Smokes_1.0", "Hormonal Contraceptives_0.0",
    "Hormonal Contraceptives_1.0", "IUD_0.0", "IUD_1.0", "STDs_0.0", "STDs_1.0",
    "Dx:Cancer_0", "Dx:Cancer_1", "Dx:CIN_0", "Dx:CIN_1", "Dx:HPV_0", "Dx:HPV_1",
    "Dx_0", "Dx_1", "Hinselmann_0", "Hinselmann_1", "Citology_0", "Citology_1", "Schiller_0", "Schiller_1"
]

def convert_value(value, is_one_hot=False):
    if value is None or value.strip() == "":
        return False if is_one_hot else 0.0
    try:
        if isinstance(value, str):
            if value.lower() == "yes" or value == "1":
                return True if is_one_hot else 1.0
            if value.lower() == "no" or value == "0":
                return False if is_one_hot else 0.0
        return float(value)
    except ValueError:
        return False if is_one_hot else 0.0

def normalize_column_name(column_name):
    column_mapping = {
        "age": "Age",
        "num_sexual_partners": "Number of sexual partners",
        "first_sexual_intercourse": "First sexual intercourse",
        "num_pregnancies": "Num of pregnancies",
        "smokes_years": "Smokes (years)",
        "smokes_packs": "Smokes (packs/year)",
        "hormonal_contraceptives_years": "Hormonal Contraceptives (years)",
        "iud_years": "IUD (years)",
        "stds_number": "STDs (number)",
        "stds_condylomatosis": "STDs:condylomatosis",
        "stds_vaginal_condylomatosis": "STDs:vaginal condylomatosis",
        "stds_vulvo_perineal_condylomatosis": "STDs:vulvo-perineal condylomatosis",
        "stds_syphilis": "STDs:syphilis",
        "stds_pelvic_inflammatory_disease": "STDs:pelvic inflammatory disease",
        "stds_genital_herpes": "STDs:genital herpes",
        "stds_molluscum_contagiosum": "STDs:molluscum contagiosum",
        "stds_hiv": "STDs:HIV",
        "stds_hepatitis_b": "STDs:Hepatitis B",
        "stds_hpv": "STDs:HPV",
        "stds_number_of_diagnosis": "STDs: Number of diagnosis",
        "Dx:Cancer_0.0": "Dx:Cancer_0",
        "Dx:Cancer_1.0": "Dx:Cancer_1",
        "Dx:CIN_0.0": "Dx:CIN_0",
        "Dx:CIN_1.0": "Dx:CIN_1",
        "Dx:HPV_0.0": "Dx:HPV_0",
        "Dx:HPV_1.0": "Dx:HPV_1",
        "Dx_0.0": "Dx_0",
        "Dx_1.0": "Dx_1",
        "Hinselmann_0.0": "Hinselmann_0",
        "Hinselmann_1.0": "Hinselmann_1",
        "Schiller_0.0": "Schiller_0",
        "Schiller_1.0": "Schiller_1",
        "Citology_0.0": "Citology_0",
        "Citology_1.0": "Citology_1"
    }
    return column_mapping.get(column_name, column_name)
def predict_risk(input_data, model, scaler, train_columns):
    input_data = pd.DataFrame([input_data])[train_columns]
    input_data_scaled = scaler.transform(input_data)
    probabilities = model.predict_proba(input_data_scaled)
    prediction = 1 if probabilities[0][1] >= 0.5 else 0
    return prediction

def predict_risk_2(input_data, model, scaler, train_columns):
    input_data = pd.DataFrame([input_data])[train_columns]
    input_data_scaled = scaler.transform(input_data)
    # probabilities = model.predict_proba(input_data_scaled)
    # prediction = 1 if probabilities[0][1] >= 0.5 else 0
    result = model.predict(input_data_scaled)
    prediction = (result > 0.5).astype("int32").flatten()
    return prediction

def calculate_statistics(results):
    total_rows = len(results)
    prediction_0 = sum(1 for r in results if r['prediction'] == 0)
    prediction_1 = total_rows - prediction_0
    biopsy_0 = sum(1 for r in results if r['biopsy'] == '0')
    biopsy_1 = total_rows - biopsy_0
    same = sum(1 for r in results if r['prediction'] == r['biopsy'])
    prediction_0_biopsy_1 = sum(1 for r in results if r['prediction'] == 0 and r['biopsy'] == '1')
    prediction_1_biopsy_0 = sum(1 for r in results if r['prediction'] == 1 and r['biopsy'] == '0')

    statistics = {
        'total_rows': total_rows,
        'prediction_0': prediction_0,
        'prediction_1': prediction_1,
        'biopsy_0': biopsy_0,
        'biopsy_1': biopsy_1,
        'same': same,
        'prediction_0_biopsy_1': prediction_0_biopsy_1,
        'prediction_1_biopsy_0': prediction_1_biopsy_0,
    }
    return statistics
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            form_data = request.form.to_dict()
            processed_data = {}
            one_hot_columns = [
                'smokes', 'hormonal_contraceptives', 'iud', 'stds',
                'dx_cancer', 'dx_cin', 'dx_hpv', 'dx',
                'Hinselmann', 'Schiller', 'Citology'
            ]
            for col, value in form_data.items():
                if col in one_hot_columns:
                    value = convert_value(value, is_one_hot=True)
                    processed_data[f"{col}_0.0"] = not value
                    processed_data[f"{col}_1.0"] = value
                else:
                    processed_data[col] = convert_value(value)

            normalized_data = {normalize_column_name(col): value for col, value in processed_data.items()}
            for col in train_columns:
                if col not in normalized_data:
                    normalized_data[col] = False if "_0" in col or "_1" in col else 0

            prediction = predict_risk_2(normalized_data, model, scaler, train_columns)

        except Exception as e:
            prediction = f"Lỗi xử lý: {str(e)}"
            print("Lỗi xử lý:", e)

    return render_template('index2.html', prediction=prediction)

@app.route('/predict_from_csv', methods=['POST'])
def predict_from_csv():
    try:
        csv_file = request.files['csv_file']
        csv_data = csv.DictReader(csv_file.stream.read().decode('utf-8').splitlines())
        results = []
            
        for row in csv_data:
            biopsy_value = row.get('Biopsy')  # Lưu giá trị Biopsy
            processed_data = {}
            one_hot_columns = [
                'Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs',
                'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx',
                'Hinselmann', 'Schiller', 'Citology'
            ]
            for col, value in row.items():
                if col == 'Biopsy':  # Bỏ qua cột Biopsy khi xử lý dữ liệu cho model
                    continue
                if col in one_hot_columns:
                    value = convert_value(value, is_one_hot=True)
                    processed_data[f"{col}_0.0"] = not value
                    processed_data[f"{col}_1.0"] = value
                else:
                    processed_data[col] = convert_value(value)

            normalized_data = {normalize_column_name(col): value for col, value in processed_data.items()}
            for col in train_columns:
                if col not in normalized_data:
                    normalized_data[col] = False if "_0" in col or "_1" in col else 0

            prediction = predict_risk_2(normalized_data, model, scaler, train_columns)
            results.append({"data": row, "prediction": prediction, "biopsy": biopsy_value})
        statistics = calculate_statistics(results)
        return render_template('results.html', results=results,statistics=statistics)  

    except Exception as e:
        return f"Lỗi: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)