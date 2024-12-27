import flwr as fl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import get_peft_model, LoraConfig
from peft import PeftModel
from huggingface_hub import HfApi

# Load the base model (BioMistral-7B in this case)
model_name = "BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM"
base_model = AutoModelForCausalLM.from_pretrained(model_name)

# Load PEFT (LoRA) adapter configuration and apply to the model
lora_config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.1, task_type="CAUSAL_LM")
model = get_peft_model(base_model, lora_config)

# Federated Averaging Strategy with Model Save Callback and PEFT Integration
class HuggingFaceFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, huggingface_save_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.huggingface_save_name = huggingface_save_name

    def aggregate_fit(self, server_round, results, failures):
        # Perform the standard FedAvg aggregation
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Load aggregated parameters into the PEFT model
            state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), aggregated_parameters)}
            self.model.load_state_dict(state_dict, strict=True)

            # Save the model locally
            save_name = f"{self.huggingface_save_name}-round-{server_round}"
            self.model.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)

            # Push model to Hugging Face
            self.push_to_huggingface(save_name)

            print(f"Model saved and pushed to Hugging Face: {save_name}")

        return aggregated_parameters

    def push_to_huggingface(self, model_dir):
        # Use Hugging Face API to upload the model
        api = HfApi()
        repo_name = self.huggingface_save_name  # Your Hugging Face repository name

        try:
            api.upload_folder(
                folder_path=model_dir,  # Path to the folder with the model
                repo_id=repo_name,  # Replace with your repo ID
                commit_message=f"Update model after federated learning round"
            )
            print(f"Successfully pushed the model to Hugging Face repository: {repo_name}")
        except Exception as e:
            print(f"Error pushing model to Hugging Face: {str(e)}")


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Start the server with the custom strategy
strategy = HuggingFaceFedAvg(
    model=model,
    huggingface_save_name="MedFusion-A-Federated-Finetuned-LLM", 
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
)

# Run server with strategy
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
