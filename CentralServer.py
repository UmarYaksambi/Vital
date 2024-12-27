import flwr as fl
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the initial model (optional for server)
model_name = "BioMistral/BioMistral-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Federated Averaging Strategy with Model Save Callback
class HuggingFaceFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model, huggingface_save_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.huggingface_save_name = huggingface_save_name

    def aggregate_fit(self, server_round, results, failures):
        # Perform the standard FedAvg aggregation
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Load aggregated parameters into the model
            state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), aggregated_parameters)}
            self.model.load_state_dict(state_dict, strict=True)

            # Save the model to Hugging Face
            save_name = f"{self.huggingface_save_name}-round-{server_round}"
            self.model.save_pretrained(save_name)
            tokenizer.save_pretrained(save_name)

            print(f"Model saved to Hugging Face: {save_name}")

        return aggregated_parameters


# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Start the server with the custom strategy
strategy = HuggingFaceFedAvg(
    model=model,
    huggingface_save_name="personalized-healthcare-llm",  # Replace with your Hugging Face repository name
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)
