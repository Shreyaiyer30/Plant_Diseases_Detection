import sqlite3
import json
import re

class PlantCareChatbot:
    def __init__(self, db_path):
        self.db_path = db_path
        
    def get_response(self, user_query, disease_name):
        """
        Generate response using:
        1. Intent matching (greetings, thanks, treatment, prevention, chemicals, organic)
        2. Keyword extraction from user query
        3. Database lookup for relevant information
        4. Fallback responses
        """
        intent = self.extract_intent(user_query)
        data = self.get_treatment_data(disease_name)
        
        if not data:
            return {
                "response": f"I'm sorry, I don't have specific treatment information for {disease_name} yet. However, generally ensuring proper watering and removing infected leaves can help.",
                "sources": []
            }
            
        response, sources = self.format_response(intent, data, disease_name)
        return {
            "response": response,
            "sources": sources
        }
        
    def extract_intent(self, query):
        query = query.lower()
        if any(word in query for word in ['hi', 'hello', 'hey', 'greetings']):
            return 'greeting'
        if any(word in query for word in ['thank', 'thanks', 'bye']):
            return 'farewell'
        if any(word in query for word in ['prevent', 'prevention', 'stop', 'avoid']):
            return 'prevention'
        if any(word in query for word in ['chemical', 'medicine', 'pesticide', 'fungicide']):
            return 'chemical'
        if any(word in query for word in ['organic', 'natural', 'home', 'homemade']):
            return 'organic'
        if any(word in query for word in ['treat', 'cure', 'fix', 'how to', 'steps']):
            return 'treatment'
        if any(word in query for word in ['symptom', 'look like', 'signs']):
            return 'symptoms'
        if any(word in query for word in ['cause', 'why', 'reason']):
            return 'cause'
        return 'unknown'
        
    def get_treatment_data(self, disease_name):
        # Fetch from database
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM plant_treatments WHERE disease_name = ?", (disease_name,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
            return None
        except Exception as e:
            print(f"Database error: {e}")
            return None
        
    def format_response(self, intent, data, disease_name):
        # Format user-friendly responses
        sources = []
        response = ""
        
        disease_clean = disease_name.replace('_', ' ')
        
        if intent == 'greeting':
            response = f"Hello! I see your plant may have {disease_clean}. How can I help you treat it today?"
        elif intent == 'farewell':
            response = "You're welcome! I hope your plant gets better soon. Feel free to ask more questions anytime."
        elif intent == 'treatment':
            response = f"🌿 TREATMENT STEPS for {disease_clean}:\n{data['treatment_steps']}\n\n⚠️ Apply treatment early morning or evening for best results."
            sources.append("treatment_steps")
        elif intent == 'prevention':
            response = f"🛡️ PREVENTION TIPS for {disease_clean}:\n{data['prevention_tips']}\n\nConsistent prevention is 90% of plant healthcare!"
            sources.append("prevention_tips")
        elif intent == 'organic':
            response = f"🌱 ORGANIC REMEDIES for {disease_clean}:\n{data['organic_remedies']}"
            sources.append("organic_remedies")
        elif intent == 'chemical':
            response = f"⚗️ CHEMICAL OPTIONS for {disease_clean}:\n{data['chemical_remedies']}"
            sources.append("chemical_remedies")
        else:
            # General fallback if intent is unknown but we have data
            response = f"I'm here to help with {disease_clean}. You can ask about treatment steps, prevention tips, or remedies (organic/chemical)."
            
        return response, sources
