from flask import Flask, render_template, request
from owlready2 import *
from transformers import pipeline

app = Flask(__name__)

# Load the ontology
onto = get_ontology("shapes_formula.rdf").load()

# Initialize the NLP model for mapping user input
try:
    nlp_model = pipeline('text-classification', model='distilbert-base-uncased-finetuned-sst-2-english')  # Default Hugging Face model
except Exception as e:
    nlp_model = None
    print(f"Error loading NLP model: {e}")

# Define the fallback synonyms
SYNONYMS = {
    "circle": "circle",
    "rectangle": "rectangle",
    "triangle": "triangle",
    "square": "square",
    "area of circle": "circle",
    "area of rectangle": "rectangle",
    "perimeter of rectangle": "rectangle",
    "perimeter of square": "square",
    "circumference of circle": "circle"
}

def validate_ai_output(label):
    """
    Validate if the AI-mapped label exists in the ontology or synonyms.
    """
    # Get all valid ontology labels
    valid_labels = []
    for individual in onto.individuals():
        valid_labels.extend([lbl.lower() for lbl in getattr(individual, "label", [])])

    # Check if the label is valid
    return label in valid_labels or label in SYNONYMS

def ai_map_concept(concept):
    """
    Uses an AI model or fallback synonyms to map user input to normalized ontology concepts.
    """
    if nlp_model:
        try:
            prediction = nlp_model(concept)
            label = prediction[0]['label'].lower()
            print(f"AI-mapped label: {label}")  # Debugging line

            # Check if the AI label is valid within synonyms or ontology
            if validate_ai_output(label):
                return SYNONYMS.get(label, label)

            # Fallback to user input normalization if AI prediction is invalid
            print(f"AI mapping failed, falling back to synonyms for: {concept}")
        except Exception as e:
            print(f"AI mapping error: {e}")

    # Fallback: Use synonym dictionary for input normalization
    normalized_concept = concept.strip().lower()
    print(f"Fallback normalized concept: {normalized_concept}")  # Debugging line
    return SYNONYMS.get(normalized_concept, normalized_concept)

def query_ontology(concept):
    """
    Query the ontology to retrieve formulas based on a user-provided concept.
    """
    # Normalize the input concept using AI or fallback
    normalized_concept = ai_map_concept(concept)
    print(f"Normalized concept: {normalized_concept}")  # Debugging line

    # Iterate over ontology individuals to find a match
    for individual in onto.individuals():
        # Get all labels for the individual in lowercase
        individual_labels = [lbl.lower() for lbl in getattr(individual, "label", [])]
        print(f"Checking individual labels: {individual_labels}")  # Debugging line

        if normalized_concept in individual_labels:
            # Fetch formulas
            area_formula = getattr(individual, "hasAreaFormula", None)
            perimeter_formula = getattr(individual, "hasPerimeterFormula", None)

            # Process formulas for output
            area = (area_formula[0].strip() if isinstance(area_formula, list) else area_formula.strip()) if area_formula else "Area formula not available."
            perimeter = (perimeter_formula[0].strip() if isinstance(perimeter_formula, list) else perimeter_formula.strip()) if perimeter_formula else None

            # Generate response
            response = f"Area Formula: {area}"
            if perimeter:
                response += f"<br>Perimeter/Circumference Formula: {perimeter}"

            return response

    return "The concept you entered was not found. Please try again."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/shapes', methods=['GET', 'POST'])
def shapes():
    if request.method == 'POST':
        concept = request.form['concept']
        result = query_ontology(concept)
        return render_template('shapes.html', concept=concept, response=result)
    return render_template('shapes.html', concept=None, response=None)

@app.route('/debug_ontology')
def debug_ontology():
    individuals = []
    for individual in onto.individuals():
        labels = getattr(individual, "label", [])
        area_formula = getattr(individual, "hasAreaFormula", None)
        perimeter_formula = getattr(individual, "hasPerimeterFormula", None)
        individuals.append({
            "labels": labels,
            "area_formula": area_formula,
            "perimeter_formula": perimeter_formula
        })
    return {"ontology_individuals": individuals}

if __name__ == "__main__":
    app.run(debug=True)
