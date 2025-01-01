import json
import flwr as fl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import sys

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token  # Fix for missing padding token
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", load_in_8bit=False, device_map="auto")

def load_local_data():
    with open("data/patient_data.json", "r") as f:
        patient_data = json.load(f)
        print(patient_data)

    with open("data/doctor_diagnosis.json", "r") as f:
        doctor_diagnosis = json.load(f)

    with open("data/llm_diagnosis.json", "r") as f:
        llm_diagnosis = json.load(f)

    # Merge data by patient ID
    merged_data = []
    for patient in patient_data:
        llm_diag = next((d for d in llm_diagnosis if d["patient_id"] == patient["id"]), {})
        doctor_diag = next((d for d in doctor_diagnosis if d["patient_id"] == patient["id"]), {})
        merged_data.append({**patient, **llm_diag, **doctor_diag})
    print('in the local load data fn')
    print(merged_data)
    print(type(merged_data))
    if type(merged_data) is None:
        sys.exit(0)
    return merged_data

def prepare_training_data(data):
    examples = []
    for entry in data:
        input_text = f"""
        Age: {entry['age']}, History: {entry['medical_history']},
        Family History: {entry['family_history']}, Symptoms: {entry['symptoms']}
        LLM Diagnosis: {entry.get('diagnosis', 'N/A')}, Treatment Plan: {entry.get('treatment_plan', 'N/A')},
        Medications: {entry.get('medication', 'N/A')}, Lifestyle Changes: {entry.get('lifestyle_changes', 'N/A')}
        """

        target_text = f"""
        Doctor's Notes: {entry.get('actual_diagnosis', 'N/A')}, Medication: {entry.get('doctor_medication', 'N/A')},
        Treatment Plan: {entry.get('doctor_treatment_plan', 'N/A')}, Lifestyle Changes: {entry.get('doctor_lifestyle_changes', 'N/A')}
        Outcome: {entry.get('outcome', 'N/A')}
        """
        examples.append((input_text, target_text))
    return examples

def tokenize_data(examples):
    inputs = tokenizer([ex[0] for ex in examples], padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer([ex[1] for ex in examples], padding="max_length", truncation=True, return_tensors="pt")
    return inputs, labels

# Custom Dataset Class
class MedicalDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.inputs["input_ids"][idx],
            "attention_mask": self.inputs["attention_mask"][idx],
            "labels": self.labels["input_ids"][idx],
        }


class HealthcareClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model=model
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(model, self.lora_config)
        # self.local_data = load_local_data()
        # if self.local_data== None:    
            # print('fuck')
            # sys.exit(0)
        # self.train_examples = prepare_training_data(self.local_data)
        k = load_local_data()
        print(type(k))
        if type(k) is None:
            sys.exit(0)
        self.train_examples = prepare_training_data(k)
        
        self.inputs, self.labels = tokenize_data(self.train_examples)
        self.dataset = MedicalDataset(self.inputs, self.labels)

    # def get_parameters(self):
    #     return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)


fl.client.start_client(
        server_address="127.0.0.1:8000", 
        client=HealthcareClient(model), 
        grpc_max_message_length = 1024*1024*1024
)

