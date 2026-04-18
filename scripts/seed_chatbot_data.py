import sqlite3
import os
import json

KAGGLE_MAP = {
    "Pepper__bell___Bacterial_spot": "Bacterial Leaf Spot",
    "Tomato_Early_blight": "Tomato - Early Blight",
    "Tomato_Late_blight": "Tomato - Late Blight",
    "Tomato_Leaf_Mold": "Powdery Mildew",
    "Tomato_Septoria_leaf_spot": "Bacterial Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Bacterial Leaf Spot",
    "Tomato__Target_Spot": "Tomato - Early Blight",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato - Leaf Curl Virus",
    "Potato___Early_blight": "Potato - Early Blight",
    "Potato___Late_blight": "Potato - Late Blight",
    "Apple___Cedar_apple_rust": "Rust",
    "Corn_(maize)___Common_rust_": "Rust",
}

def seed_data():
    db_path = os.path.join('instance', 'plantcure.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS plant_treatments (
        id INTEGER PRIMARY KEY,
        disease_name TEXT UNIQUE,
        treatment_steps TEXT,
        prevention_tips TEXT,
        chemical_remedies TEXT,
        organic_remedies TEXT,
        common_questions TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    diseases = [
        {
            "name": "Pepper__bell___Bacterial_spot",
            "treatment": "1. Remove and destroy infected plant parts.\n2. Avoid overhead watering to reduce spread.\n3. Apply copper-based fungicides if necessary.",
            "prevention": "• Use disease-free seeds.\n• Rotate crops every 2-3 years.\n• Avoid working in the garden when plants are wet.",
            "organic": "• Neem oil spray.\n• Baking soda solution (1 tsp per quart of water).\n• Garlic-pepper spray.",
            "chemical": "• Copper fungicides.\n• Streptomycin (if permitted in your area)."
        },
        {
            "name": "Tomato_Early_blight",
            "treatment": "1. Prune lower leaves to improve airflow.\n2. Apply mulching to prevent soil splash.\n3. Use fungicides containing chlorothalonil or copper.",
            "prevention": "• Practice crop rotation.\n• Use resistant varieties.\n• Ensure proper spacing between plants.",
            "organic": "• Compost tea.\n• Bacillus subtilis based bio-fungicides.\n• Serenade Garden Disease Control.",
            "chemical": "• Chlorothalonil.\n• Mancozeb.\n• Copper-based fungicides."
        },
        {
            "name": "Tomato_Late_blight",
            "treatment": "1. Remove and bag infected plants immediately.\n2. Do not compost infected material.\n3. Apply preventative fungicides to nearby healthy plants.",
            "prevention": "• Avoid overhead irrigation.\n• Keep foliage dry.\n• Monitor weather conditions (wet/cool favors blight).",
            "organic": "• Fixed copper products.\n• Potassium bicarbonate sprays.\n• Trichoderma harzianum.",
            "chemical": "• Chlorothalonil.\n• Ridomil Gold.\n• Copper sulfate."
        },
        {
            "name": "Tomato_Leaf_Mold",
            "treatment": "1. Increase ventilation in greenhouses.\n2. Reduce humidity levels.\n3. Remove infected leaves to stop spore spread.",
            "prevention": "• Use resistant cultivars.\n• Space plants appropriately.\n• Avoid wetting leaves during watering.",
            "organic": "• Vinegar spray (mild solution).\n• Milk-water spray (40/60 ratio).\n• Neem oil.",
            "chemical": "• Difenoconazole.\n• Chlorothalonil."
        },
        {
            "name": "Tomato_Septoria_leaf_spot",
            "treatment": "1. Remove affected leaves as soon as they appear.\n2. Improve air circulation.\n3. Apply fungicides every 7-10 days.",
            "prevention": "• Remove garden debris in autumn.\n• Water at the base of the plant.\n• Use stakes or cages to keep plants off the ground.",
            "organic": "• Copper-based soap.\n• Baking soda and horticultural oil mixture.",
            "chemical": "• Penthiopyrad.\n• Chlorothalonil."
        },
        {
            "name": "Tomato_Spider_mites_Two_spotted_spider_mite",
            "treatment": "1. Blast plants with a strong stream of water.\n2. Introduce natural predators like ladybugs.\n3. Apply insecticidal soap or oil.",
            "prevention": "• Keep plants well-hydrated.\n• Increase humidity.\n• Avoid excessive nitrogen fertilizers.",
            "organic": "• Rosemary oil.\n• Insecticidal soap.\n• Diatomaceous earth.",
            "chemical": "• Abamectin.\n• Bifenazate."
        },
        {
            "name": "Tomato__Target_Spot",
            "treatment": "1. Remove infected plant material.\n2. Improve drainage and airflow.\n3. Use protectant fungicides.",
            "prevention": "• Avoid overhead irrigation.\n• Space plants for air movement.\n• Manage weeds around the area.",
            "organic": "• Neem oil.\n• Potassium bicarbonate.",
            "chemical": "• Chlorothalonil.\n• Azoxystrobin."
        },
        {
            "name": "Tomato__Tomato_YellowLeaf__Curl_Virus",
            "treatment": "1. Remove and destroy infected plants immediately.\n2. Control whitefly populations (the primary vector).\n3. Use reflective mulches to repel whiteflies.",
            "prevention": "• Use virus-resistant varieties.\n• Install fine-mesh screens in greenhouses.",
            "organic": "• Yellow sticky traps for whiteflies.\n• Neem oil for vector control.",
            "chemical": "• Imidacloprid (for whiteflies).\n• Dinotefuran."
        },
        {
            "name": "Potato___Early_blight",
            "treatment": "1. Maintain plant vigor with proper fertilization.\n2. Harvest as soon as tubers are mature.\n3. Apply fungicides selectively.",
            "prevention": "• Avoid planting potatoes near tomatoes.\n• Rotate with non-host crops.\n• Use certified disease-free tubers.",
            "organic": "• Bacillus amyloliquefaciens.\n• Compost mulching.",
            "chemical": "• Chlorothalonil.\n• Mancozeb."
        },
        {
            "name": "Potato___Late_blight",
            "treatment": "1. Destroy all 'volunteer' potatoes in the spring.\n2. Apply fungicides early in the season.\n3. Kill vines before harvest if blight is present.",
            "prevention": "• Use resistant varieties.\n• Ensure deep hilling of potatoes.\n• Monitor BLIGHTLINE forecasts.",
            "organic": "• Copper-based sprays.\n• Serenade (bio-fungicide).",
            "chemical": "• Fluazinam.\n• Cymoxanil."
        },
        {
            "name": "Rust",
            "treatment": "1. Remove and destroy heavily infected leaves.\n2. Apply sulfur or copper-based fungicides at the first sign of pustules.\n3. Ensure plants have adequate spacing for airflow.",
            "prevention": "• Plant resistant varieties if available.\n• Avoid overhead watering that keeps leaves wet.\n• Remove fallen leaves and debris around plants in autumn.",
            "organic": "• Neem oil spray.\n• Sulfur dust or spray.\n• Compost tea to boost plant immunity.",
            "chemical": "• Chlorothalonil.\n• Myclobutanil.\n• Mancozeb."
        }
    ]

    for d in diseases:
        names = {d['name']}
        mapped = KAGGLE_MAP.get(d['name'])
        if mapped:
            names.add(mapped)

        for disease_name in names:
            cursor.execute('''
            INSERT OR REPLACE INTO plant_treatments 
            (disease_name, treatment_steps, prevention_tips, chemical_remedies, organic_remedies, common_questions)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (disease_name, d['treatment'], d['prevention'], d['chemical'], d['organic'], json.dumps([])))

    conn.commit()
    conn.close()
    print("Database seeded successfully with Rust data!")

if __name__ == "__main__":
    seed_data()
