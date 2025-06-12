# Fine-tuning FinBERT for Financial Sentiment Classification in Google Colab with LoRA and Export Options

# Step 1: Install required packages (for Google Colab users only)
# Uncomment and run in Colab
# %pip install -q transformers datasets peft accelerate bitsandbytes wandb scikit-learn

# Step 2: Import required libraries
import wandb
from transformers import AutoTokenizer, TrainingArguments, Trainer, EvalPrediction
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os # Import the os module

# Step 3: Login to Weights & Biases
# wandb.login() # Commented out for testing purposes, uncomment if you want to log to wandb

# Step 4: Load financial domain-specific model and tokenizer
model_id = "ProsusAI/finbert"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,  # FinBERT has 3 sentiment classes: positive, neutral, negative
)

# Add this before lora_config to debug available modules
for name, module in base_model.named_modules():
    print(name)

print(f"Loaded model {model_id} with {base_model.num_parameters()} parameters.")
print("Model hidden size: {base_model.config.hidden_size}")

# Get various model dimensions
print(f"Hidden size: {base_model.config.hidden_size}")
print(f"Number of attention heads: {base_model.config.num_attention_heads}")
print(f"Number of hidden layers: {base_model.config.num_hidden_layers}")
print(f"Intermediate size: {base_model.config.intermediate_size}")

# Step 5: Apply LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],  # BERT attention layer names
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS
)

model = get_peft_model(base_model, lora_config)

# After model = get_peft_model(base_model, lora_config)
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.numel()}")

model.print_trainable_parameters()

# Step 6: Load and preprocess dataset (expects JSON with 'text' and 'label' fields)

# Print current working directory to verify file location
print(f"Current working directory: {os.getcwd()}")

# Check if files exist
train_file = "train.json"
val_file = "val.json"

if not os.path.exists(train_file):
    print(f"Error: {train_file} not found in {os.getcwd()}")
elif not os.path.exists(val_file):
    print(f"Error: {val_file} not found in {os.getcwd()}")
else:
    print(f"Found {train_file} and {val_file}.")
    try:
        dataset = load_dataset("json", data_files={"train": train_file, "validation": val_file})
        print("Dataset loaded successfully.")

        def tokenize(example):
            return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

        tokenized_dataset = dataset.map(tokenize)

        # Step 7: Set training arguments
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")

        training_args = TrainingArguments(
            output_dir="./finbert_seq_cls",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=3,
            eval_strategy="steps",        # Updated from evaluation_strategy
            save_strategy="steps",
            eval_steps=100,
            save_steps=100,
            learning_rate=2e-5,
            logging_steps=20,
            fp16=False,                   # Disabled for MPS compatibility
            report_to="wandb",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1"
        )

        # Step 8: Define metrics for evaluation
        def compute_metrics(p: EvalPrediction):
            preds = np.argmax(p.predictions, axis=1)
            acc = accuracy_score(p.label_ids, preds)
            f1 = f1_score(p.label_ids, preds, average="macro")
            return {"accuracy": acc, "f1": f1}

        # Step 9: Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            compute_metrics=compute_metrics
        )

        # Step 10: Start training
        trainer.train()

        # Step 11: Save model and tokenizer
        model.save_pretrained("./finbert_seq_cls")
        tokenizer.save_pretrained("./finbert_seq_cls")

        # Step 12: Modified TorchScript export
        try:
            # Create a wrapper class for inference
            class FinBERTWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, input_ids):
                    outputs = self.model(input_ids)
                    return outputs.logits  # Return only logits tensor

            # Prepare example input
            example_input = tokenizer(
                "The stock surged after the positive earnings report.", 
                return_tensors="pt"
            )
            input_ids = example_input["input_ids"].to(device)

            # Wrap model and prepare for export
            model_cpu = model.cpu()
            wrapped_model = FinBERTWrapper(model_cpu)
            wrapped_model.eval()  # Set to evaluation mode

            # Export with wrapper
            with torch.no_grad():
                torch_script_model = torch.jit.trace(
                    wrapped_model,
                    input_ids.cpu(),
                    strict=False  # Allow flexible tracing
                )
                torch.jit.save(torch_script_model, "./finbert_seq_cls/model_torchscript.pt")
                print("Model successfully exported to TorchScript")

        except Exception as e:
            print(f"Error during TorchScript export: {e}")

        # Optional - Export to ONNX (uncomment below if needed and ensure required packages are installed)
        # from transformers.onnx import export
        # from transformers.onnx.features import FeaturesManager
        # model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="sequence-classification")
        # onnx_config = model_onnx_config(model.config)
        # export(tokenizer, model, onnx_config, opset=13, output="./finbert_seq_cls/model.onnx")

    except Exception as e:
        print(f"An error occurred during dataset loading or processing: {e}")

# Step 13: Model Testing
def predict_sentiment(text, model, tokenizer, device):
    # Move model to specified device
    model = model.to(device)
    
    # Prepare input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Move input tensors to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
    # Convert to numpy for easier handling
    logits = logits.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Get predicted class
    predicted_class = np.argmax(logits, axis=1)[0]
    
    # Map to sentiment labels
    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    
    return {
        "text": text,
        "sentiment": sentiment_map[predicted_class],
        "logits": logits[0],
        "probabilities": probs[0]
    }

# Before testing, ensure model is on correct device
model = model.to(device)

# Test the model with example texts
test_texts = [
    "The company reported strong earnings, beating market expectations.",
    "Stock prices remained unchanged in today's trading session.",
    "The company's shares plummeted after the earnings miss."
]

print("\nTesting the model:")
for text in test_texts:
    result = predict_sentiment(text, model, tokenizer, device)
    print(f"\nText: {result['text']}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Logits: {result['logits']}")
    print(f"Probabilities: {result['probabilities']}")