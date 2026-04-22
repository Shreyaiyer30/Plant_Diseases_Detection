# test_translate.py
from deep_translate_helper import translate_disease_name, translate_text

# Test the translations
test_diseases = [
    "Corn_(maize)___Common_rust_",
    "Tomato___Early_Blight",
    "Apple___Apple_scab",
    "Healthy"
]

print("Testing Hindi translations:")
for d in test_diseases:
    result = translate_disease_name(d, 'hi')
    print(f"  {d} -> {result}")

print("\nTesting Marathi translations:")
for d in test_diseases:
    result = translate_disease_name(d, 'mr')
    print(f"  {d} -> {result}")