# Bayesian network CPTs
CPT = {
    'B': {'+b': 0.001, '-b': 0.999},
    'E': {'+e': 0.002, '-e': 0.998},
    'A': {
        ('+b', '+e'): {"+a": 0.95, "-a": 0.05},
        ('+b', '-e'): {"+a": 0.94, "-a": 0.06},
        ('-b', '+e'): {"+a": 0.29, "-a": 0.71},
        ('-b', '-e'): {"+a": 0.001, "-a": 0.999},
    },
    'J': {"+a": {"+j": 0.9, "-j": 0.1}, "-a": {"+j": 0.05, "-j": 0.95}},
    'M': {"+a": {"+m": 0.7, "-m": 0.3}, "-a": {"+m": 0.01, "-m": 0.99}},
}

# Bayesian network structure
structure = {
    'B': [],
    'E': [],
    'A': ['B', 'E'],
    'J': ['A'],
    'M': ['A'],
}

def flatten_cpt(cpt):
    """Flatten a nested CPT into a single dictionary mapping assignments to probabilities."""
    flat_cpt = {}
    for key, value in cpt.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                combined_key = key + (sub_key,) if isinstance(key, tuple) else (key, sub_key)
                flat_cpt[combined_key] = sub_value
        else:
            flat_cpt[(key,)] = value
    return flat_cpt

def factor_product(f1, f2):
    """Multiply two factors, ensuring consistency of variable assignments."""
    
    result = {}
    for assignment1, prob1 in f1.items():
        for assignment2, prob2 in f2.items():
            combined = dict(assignment1)  
            combined.update(dict(assignment2))  
            if len(combined) == len(set(combined.keys())):
                result[tuple(sorted(combined.items()))] = prob1 * prob2
    return result

def marginalize(factor, var):
    """Marginalize a variable from a factor."""
    
    new_factor = {}
    for assignment, prob in factor.items():
        reduced_assignment = tuple((k, v) for k, v in assignment if k != var)
        new_factor[reduced_assignment] = new_factor.get(reduced_assignment, 0) + prob

    return new_factor

def apply_evidence(factor, evidence):
    """Apply evidence to a factor."""
    new_factor = {}
    for assignment, prob in factor.items():
        valid = True
        for var, val in evidence.items():
            if any(k == var and v != val for k, v in assignment):
                valid = False
                break
        if valid:
            new_factor[assignment] = prob
    return new_factor

def normalize(factor):
    """Normalize the factor so that probabilities sum to 1."""
    total = sum(factor.values())
    normalized_factor = {k: v / total for k, v in factor.items()}
    return normalized_factor

def variable_elimination(query, evidence, hidden_vars):
    """Variable elimination for Bayesian networks."""
    print("\n--- Starting Variable Elimination ---")

    factors = {var: flatten_cpt(CPT[var]) for var in structure.keys()}
    print()

    print("Applying Evidence...")
    for var, val in evidence.items():
        for factor_var in list(factors.keys()):
            factors[factor_var] = apply_evidence(factors[factor_var], {var: val})

    print("\nEliminating Hidden Variables:")
    for var in hidden_vars:
        if var == query or var in evidence:
            continue  
        
        relevant_factors = [
        factors.pop(key) for key in list(factors.keys())
        if var in key  
        ]

        if not relevant_factors:
            continue

        combined_factor = relevant_factors[0]
        for f in relevant_factors[1:]:
            combined_factor = factor_product(combined_factor, f)
        
        marginalized_factor = marginalize(combined_factor, var)
        factors[var] = marginalized_factor

    print("\nCombining remaining factors:")
    final_factor = list(factors.values())[0]
    for f in list(factors.values())[1:]:
        final_factor = factor_product(final_factor, f)

    filtered_factor = {}
    for assignment, prob in final_factor.items():
        assignment_dict = dict(assignment)  
        for key, value in assignment_dict.items():
            if value.lower() == query.lower():  
                filtered_factor[(key, value)] = prob
                
    print("\nFiltered Factor for Query Variable:")
    print(filtered_factor)

    print("\nNormalising...")
    
    return normalize(filtered_factor)

def print_distribution_table(query_result, query, evidence):
    """Print the probability distribution in a table format."""
    print(f"\nProbability Distribution Table for P({query} | {evidence}):")
    print(f"{'Assignment':<15}{'Probability':<10}")
    print("-" * 25)
    for value, prob in query_result.items():
        if isinstance(value, tuple):  
            value_str = ','.join(value)
        else:
            value_str = value
        print(f"{query} = {value_str:<13}{prob:.4f}")

if __name__ == "__main__":
    # Query: P(Burglary | John Calls = +j)
    query = "B"
    evidence = {"J": "+j"}
    hidden_vars = ["E", "A", "M"]

    # other query to test
    # query = "B"
    # evidence = {"M": "+m"}  # Mary calls
    # hidden_vars = ["E", "A", "J"]  # E, A and J are now the hidden variables

    # other query to test
    # query = "B"
    # evidence = {"J": "+j", "M": "+m"}  # John calls and Mary calls
    # hidden_vars = ["E", "A"]  # E and A are now the hidden variables

    result = variable_elimination(query, evidence, hidden_vars)

    print_distribution_table(result, query, evidence)

