import json
import flwr as fl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from peft import PeftModel

# Load BioMistral model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM")
model = AutoModelForCausalLM.from_pretrained("BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM")

# ---------------------------
# 1. Load Local Patient Data
# ---------------------------
def load_local_data():
    with open("data/patient_data.json", "r") as f:
        patient_data = json.load(f)

    with open("data/llm_diagnosis.json", "r") as f:
        llm_diagnosis = json.load(f)

    with open("data/doctor_diagnosis.json", "r") as f:
        doctor_diagnosis = json.load(f)

    # Merge data by patient ID
    merged_data = []
    for patient in patient_data:
        llm_diag = next((d for d in llm_diagnosis if d["patient_id"] == patient["id"]), {})
        doctor_diag = next((d for d in doctor_diagnosis if d["patient_id"] == patient["id"]), {})
        merged_data.append({**patient, **llm_diag, **doctor_diag})

    return merged_data


# ---------------------------
# 2. Prepare Training Data
# ---------------------------
def prepare_training_data(data):
    examples = []
    for entry in data:
        # Constructing the input text with detailed patient information and LLM-generated diagnosis
        input_text = f"Age: {entry['age']}, History: {entry['medical_history']}, " \
                     f"Family History: {entry['family_history']}, Symptoms: {entry['symptoms']}\n" \
                     f"LLM Diagnosis: {entry.get('diagnosis', 'N/A')}, Treatment Plan: {entry.get('treatment_plan', 'N/A')}, " \
                     f"Medications: {entry.get('medication', 'N/A')}, Lifestyle Changes: {entry.get('lifestyle_changes', 'N/A')}\n"

        # Constructing the target text with actual doctor's diagnosis, outcome, treatment, etc.
        target_text = f"Doctor's Notes: {entry.get('actual_diagnosis', 'N/A')}, Medication: {entry.get('doctor_medication', 'N/A')}, " \
                      f"Treatment Plan: {entry.get('doctor_treatment_plan', 'N/A')}, Lifestyle Changes: {entry.get('doctor_lifestyle_changes', 'N/A')}" \
                      f"Outcome: {entry.get('outcome', 'N/A')},\n"
        
        examples.append((input_text, target_text))
    return examples


# ---------------------------
# 3. Tokenize Data for BioMistral
# ---------------------------
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


# ---------------------------
# 4. Federated Learning Client
# ---------------------------
class HealthcareClient(fl.client.NumPyClient):
    def __init__(self, model):
        # Apply LoRA (Low-Rank Adaptation) to the model for PEFT
        self.lora_config = LoraConfig(
            r=8,  # rank of the low-rank matrices
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # Target attention layers
            lora_dropout=0.1,
            bias="none",
        )
        self.model = get_peft_model(model, self.lora_config)  # Apply PEFT (LoRA)
        self.local_data = load_local_data()
        self.train_examples = prepare_training_data(self.local_data)
        self.inputs, self.labels = tokenize_data(self.train_examples)
        self.dataset = MedicalDataset(self.inputs, self.labels)

    def get_parameters(self):
        # Return the PEFT model's parameters (LoRA layers)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), parameters)}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        dataloader = DataLoader(self.dataset, batch_size=4)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)

        self.model.train()
        for epoch in range(3):
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

        # Return updated model parameters
        return self.get_parameters(), len(self.local_data), {}


# ---------------------------
# 5. Start FL Client and Connect to Server
# ---------------------------
fl.client.start_numpy_client(
    server_address="localhost:8080",  # Replace with your server's IP
    client=HealthcareClient(model),
)
