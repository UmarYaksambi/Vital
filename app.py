import streamlit as st
import json
from datetime import datetime
import groq
import os
from typing import Dict, List
import uuid

# Initialize Groq client
groq_client = groq.Groq(api_key="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")

# Constants
SYSTEM_PROMPT = """Role:
You are a medical AI assistant designed to conduct patient consultations, gather information, and provide preliminary assessments.

Responsibilities:

Initiate conversations by asking open-ended questions to understand the patient's condition.
Follow up with one relevant and specific question at a time, based on the patient's response.
Maintain a compassionate, professional, and reassuring tone throughout the conversation.
Ensure questions are simple, clear, and easy to understand, avoiding medical jargon where possible.
Ask about the severity, duration, and progression of symptoms.
Identify any underlying conditions, medications, or recent changes in health.
Summarize collected information before providing a preliminary diagnosis or suggesting next steps.
Example Interaction Flow:

Start with: "Hi, I’m here to help. Can you describe what symptoms you’re experiencing?"
Based on the response, ask: "When did these symptoms start?"
Continue with: "On a scale of 1 to 10, how would you rate the pain or discomfort?"
Adapt questions to explore relevant systems (e.g., respiratory, gastrointestinal, neurological) as needed.
Provide a summary and next steps, like: "From what you've described, this might be [condition]. I recommend [rest, hydration, visiting a doctor, etc.]."
"""

DISCLAIMER = """
**IMPORTANT MEDICAL DISCLAIMER:**

This is an AI-powered medical assistance tool for educational purposes only. 
The information provided does not constitute medical advice, diagnosis, or treatment.
Always consult with a qualified healthcare professional for medical decisions.
In case of emergency, call your local emergency services immediately.
"""

class HealthcareChat:
    def __init__(self):
        self.initialize_session_state()
        if not st.session_state.get('messages'):
            self.initiate_conversation()

    def initialize_session_state(self):
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        if 'symptoms_collected' not in st.session_state:
            st.session_state.symptoms_collected = False
        if 'diagnosis' not in st.session_state:
            st.session_state.diagnosis = None

    def initiate_conversation(self):
        if not any(msg['role'] == 'assistant' for msg in st.session_state.messages):
            initial_prompt = "Hi, I’m here to help. Can you describe what symptoms you’re experiencing?"
            st.session_state.messages.append({"role": "assistant", "content": initial_prompt})

    def save_conversation(self, symptoms_json: Dict):
        filename = f"conversations/conversation_{st.session_state.conversation_id}.json"
        os.makedirs("conversations", exist_ok=True)
        
        data = {
            "conversation_id": st.session_state.conversation_id,
            "timestamp": datetime.now().isoformat(),
            "symptoms": symptoms_json,
            "messages": st.session_state.messages
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)

    def save_diagnosis(self, diagnosis: Dict):
        filename = f"diagnoses/diagnosis_{st.session_state.conversation_id}.json"
        os.makedirs("diagnoses", exist_ok=True)
        
        with open(filename, 'w') as f:
            json.dump(diagnosis, f, indent=4)

    def get_llm_response(self, messages: List[Dict]) -> str:
        try:
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=messages,
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error communicating with Groq API: {str(e)}")
            return "I apologize, but I'm having trouble processing your request. Please try again."

    def extract_symptoms_json(self) -> Dict:
        symptoms_prompt = """
        Based on our conversation, please extract and summarize the patient's symptoms 
        and relevant medical information in a structured JSON format. Include:
        - Main symptoms
        - Duration
        - Severity
        - Related symptoms
        - Medical history (if mentioned)
        - Risk factors (if mentioned)
        Return only JSON.
        """
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *st.session_state.messages,
            {"role": "user", "content": symptoms_prompt}
        ]
        
        response = self.get_llm_response(messages)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            st.error("Failed to parse symptoms into JSON format. Please try again.")
            return {}

    def get_diagnosis(self, symptoms_json: Dict) -> Dict:
        if not symptoms_json:
            return {"error": "No symptoms were extracted."}
        diagnosis_prompt = """
        Based on the provided symptoms and medical information, please provide:
        - Possible conditions
        - Recommended next steps
        - Lifestyle changes
        - Treatment plan (if applicable)
        - Medications (if applicable)
        Return only JSON.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(symptoms_json)},
            {"role": "user", "content": diagnosis_prompt}
        ]
        
        response = self.get_llm_response(messages)
        try:
            diagnosis = json.loads(response)
            self.save_diagnosis(diagnosis)
            return diagnosis
        except json.JSONDecodeError:
            st.error("Failed to generate diagnosis. Please try again.")
            return {}

    def render_chat_interface(self):
        st.title("AI Healthcare Assistant")
        st.info(DISCLAIMER)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Describe your symptoms..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                *st.session_state.messages
            ]
            
            response = self.get_llm_response(messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        if len(st.session_state.messages) > 4 and not st.session_state.symptoms_collected:
            if st.button("Get Diagnosis"):
                with st.spinner("Analyzing symptoms and generating diagnosis..."):
                    symptoms_json = self.extract_symptoms_json()
                    st.session_state.diagnosis = self.get_diagnosis(symptoms_json)
                    self.save_conversation(symptoms_json)
                    st.session_state.symptoms_collected = True

        if st.session_state.diagnosis:
            st.subheader("Diagnosis")
            diagnosis = st.session_state.diagnosis
            st.write("### Possible Conditions")
            for condition in diagnosis.get("possible_conditions", []):
                st.write(f"- {condition}")
            
            st.write("### Recommended Next Steps")
            for step in diagnosis.get("recommended_next_steps", []):
                st.write(f"- {step}")
            
            st.write("### Lifestyle Changes")
            for change in diagnosis.get("lifestyle_changes", []):
                st.write(f"- {change}")
            
            st.write("### Treatment Plan")
            for change in diagnosis.get("treatment_plan", []):
                st.write(f"- {change}")

            st.write("### Medication")
            for change in diagnosis.get("medication", []):
                st.write(f"- {change}")


if __name__ == "__main__":
    st.set_page_config(page_title="Healthcare AI Assistant", page_icon="🏥", layout="wide")
    chat_app = HealthcareChat()
    chat_app.render_chat_interface()
