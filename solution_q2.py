import csv

def load_data(file_path):
    """
    Load the dataset from a CSV file and return it as a list of dictionaries.
    Each dictionary represents a row.
    """
    data = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file, delimiter=',')
        for row in reader:
            data.append({
                "glucose": int(row["glucose"]),
                "bloodpressure": int(row["bloodpressure"]),
                "diabetes": int(row["diabetes"]),
            })
    print(f"Successfully loaded the dataset. Number of entries: {len(data)}")
    return data

def calculate_probabilities(data):
    """
    Calculate the conditional probabilities P(glucose | diabetes) and 
    P(bloodpressure | diabetes) based on the dataset.
    """
    print("\nCalculating Prior and Conditional Probabilities...\n")
    counts = {
        "glucose": {},
        "bloodpressure": {},
        "diabetes": {"0": 0, "1": 0},
    }
    
    for row in data:
        glucose = row["glucose"]
        bp = row["bloodpressure"]
        diabetes = row["diabetes"]
        
        counts["diabetes"][str(diabetes)] += 1
        
        if glucose not in counts["glucose"]:
            counts["glucose"][glucose] = {"0": 0, "1": 0}
        counts["glucose"][glucose][str(diabetes)] += 1
        
        if bp not in counts["bloodpressure"]:
            counts["bloodpressure"][bp] = {"0": 0, "1": 0}
        counts["bloodpressure"][bp][str(diabetes)] += 1
    
    probabilities = {"glucose": {}, "bloodpressure": {}, "diabetes": {}}
    
    total_rows = len(data)
    probabilities["diabetes"]["0"] = counts["diabetes"]["0"] / total_rows
    probabilities["diabetes"]["1"] = counts["diabetes"]["1"] / total_rows
    
    for glucose, values in counts["glucose"].items():
        probabilities["glucose"][glucose] = {
            "0": values["0"] / counts["diabetes"]["0"] if counts["diabetes"]["0"] > 0 else 0,
            "1": values["1"] / counts["diabetes"]["1"] if counts["diabetes"]["1"] > 0 else 0,
        }
    
    for bp, values in counts["bloodpressure"].items():
        probabilities["bloodpressure"][bp] = {
            "0": values["0"] / counts["diabetes"]["0"] if counts["diabetes"]["0"] > 0 else 0,
            "1": values["1"] / counts["diabetes"]["1"] if counts["diabetes"]["1"] > 0 else 0,
        }
    
    return probabilities

def print_probabilities(probabilities):
    """
    Print prior and conditional probabilities.
    """
    print("2.1.1 Prior Probabilities P(Y):")
    print(f"P(Y=0): {probabilities['diabetes']['0']:.4f}")
    print(f"P(Y=1): {probabilities['diabetes']['1']:.4f}\n")

    print("2.1.2 Conditional Probabilities P(X1 | Y):")
    for glucose, probs in probabilities['glucose'].items():
        print(f"Glucose={glucose} -> P(Y=0): {probs['0']:.4f}, P(Y=1): {probs['1']:.4f}")
    print()

    print("2.1.3 Conditional Probabilities P(X2 | Y):")
    for bp, probs in probabilities['bloodpressure'].items():
        print(f"BloodPressure={bp} -> P(Y=0): {probs['0']:.4f}, P(Y=1): {probs['1']:.4f}")
    print()

def compute_lookup_table(probabilities):
    """
    Generate a lookup table for P(Y | X1, X2) using conditional probabilities.
    """
    lookup_table = {}
    print("2.2.1, 2.2.2, 2.3.1 Generating Lookup Table for P(Y | X1, X2):\n")
    for glucose in probabilities["glucose"]:
        for bp in probabilities["bloodpressure"]:
            P_Y_1 = probabilities["diabetes"]["1"] * probabilities["glucose"][glucose]["1"] * probabilities["bloodpressure"][bp]["1"]
            P_Y_0 = probabilities["diabetes"]["0"] * probabilities["glucose"][glucose]["0"] * probabilities["bloodpressure"][bp]["0"]
            total = P_Y_1 + P_Y_0
            P_Y_1_given_X = P_Y_1 / total if total > 0 else 0
            P_Y_0_given_X = P_Y_0 / total if total > 0 else 0
            lookup_table[(glucose, bp)] = {"P(Y=1)": P_Y_1_given_X, "P(Y=0)": P_Y_0_given_X}
            print(f"(Glucose={glucose}, BloodPressure={bp}) -> P(Y=1): {P_Y_1_given_X:.4f}, P(Y=0): {P_Y_0_given_X:.4f}")
    print()
    return lookup_table

def predict(glucose, bp, lookup_table):
    """
    Predict diabetes for a given glucose and blood pressure using the lookup table.
    """
    prediction = lookup_table.get((glucose, bp), {"P(Y=1)": 0.0, "P(Y=0)": 0.0})
    return 1 if prediction["P(Y=1)"] > prediction["P(Y=0)"] else 0

def evaluate(data, lookup_table):
    """
    Evaluate accuracy using the lookup table.
    """
    correct = 0
    for row in data:
        pred = predict(row["glucose"], row["bloodpressure"], lookup_table)
        if pred == row["diabetes"]:
            correct += 1
    return correct / len(data)

def main():
    file_path = "Naive-Bayes-Classification-Data.csv"
    print("Loading the dataset...\n")
    data = load_data(file_path)
    
    probabilities = calculate_probabilities(data)
    print_probabilities(probabilities)
    
    lookup_table = compute_lookup_table(probabilities)
    
    print("2.3.2 Model Accuracy:")
    accuracy = evaluate(data, lookup_table)
    print(f"Accuracy: {accuracy*100:.2f}% \n")
    
    print("2.4: Demonstration, Sample Prediction:")
    sample_instance = {"glucose": 50, "bloodpressure": 75}
    prediction = predict(sample_instance["glucose"], sample_instance["bloodpressure"], lookup_table)
    print(f"Sample instance {sample_instance} -> Predicted diabetes: {prediction}")

if __name__ == "__main__":
    main()