import os
import torch
import numpy as np
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction
)

#traindt = "Labeled_bullionvault_articles.csv"
#traindt = "yahoo_news.csv"
#traindt = "Final_reuters_articles_processed 1.csv"
#traindt = "reuters_tokenizer_ready.csv"
traindt = "Final_gold-dataset-sinha-khandait1.csv"
working_dir = "/Users/mohanpanakam/Documents/Finetuning/finetuning"

class CustomTrainer(Trainer):
    """Custom trainer with class weights support"""
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    return cm

def calculate_detailed_metrics(trainer, dataset, split_name="Validation"):
    """Calculate and print detailed metrics"""
    predictions = trainer.predict(dataset)
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=1)
    
    # Get unique classes in the data
    unique_classes = np.unique(y_true)
    
    # Create target names based on present classes
    label_names = ['positive', 'neutral', 'negative']
    present_labels = [label_names[i] for i in unique_classes]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'classification_report': classification_report(
            y_true, 
            y_pred,
            target_names=present_labels,
            labels=unique_classes,
            digits=4
        ),
        'confusion_matrix': plot_confusion_matrix(
            y_true, 
            y_pred, 
            f"{split_name} Confusion Matrix"
        )
    }
    
    # Print results with class information
    print(f"\n=== {split_name} Set Metrics ===")
    print(f"Present classes: {unique_classes}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])
    
    return metrics

def print_trainable_lora_modules(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")

    model.print_trainable_parameters()

def train_and_evaluate_fold(fold_dataset, model, tokenizer, device, fold_num, n_splits):
    """Train and evaluate a single fold"""
    try:
        # Calculate class weights using 'labels' instead of 'label'
        labels = fold_dataset['train']['labels']
        label_counts = np.bincount(labels.cpu().numpy(), minlength=3)
        print(f"Label counts in fold {fold_num}: {label_counts}")
        
        # Handle missing classes by adding a small constant
        epsilon = 1e-6
        adjusted_counts = label_counts + epsilon
        
        # Calculate class weights manually to handle missing classes
        n_samples = len(labels)
        n_classes = 3
        adjusted_weights = n_samples / (n_classes * adjusted_counts)
        print(f"Adjusted class weights for fold {fold_num}: {adjusted_weights}")
        
        class_weights = torch.tensor(adjusted_weights, dtype=torch.float).to(device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{working_dir}/finbert_fold_{fold_num}",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            num_train_epochs=5,
            learning_rate=5e-5,
            warmup_steps=500,
            weight_decay=0.1,
            logging_steps=500,
            eval_steps=1500,
            save_steps=1500,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=False  # Disabled for MPS compatibility
        )
        
        # Initialize trainer
        trainer = CustomTrainer(
            class_weights=class_weights,
            model=model,
            args=training_args,
            train_dataset=fold_dataset["train"],
            eval_dataset=fold_dataset["validation"],
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
                "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="macro")
            }
        )
        
        # Train
        print(f"\n=== Training Fold {fold_num + 1}/{n_splits} ===")
        trainer.train()
        
        # Evaluate
        metrics = {
            'train': calculate_detailed_metrics(trainer, fold_dataset['train'], "Train"),
            'validation': calculate_detailed_metrics(trainer, fold_dataset['validation'], "Validation")
        }
        
        return trainer, metrics
        
    except Exception as e:
        print(f"Error in fold {fold_num}: {e}")
        print(f"Label counts: {label_counts}")
        return None, None

def tokenize_dataset(dataset_dict, tokenizer):
    """Tokenize dataset and format for training"""
    def preprocess_function(examples):
        # Print sample of labels for debugging
        print("Sample labels:", examples["label"][:5])
        
        # Handle label mapping for different label formats
        def map_label(label):
            # Convert to int if string
            if isinstance(label, str):
                label = int(label)
            # Only map if -1 is present in the dataset
            if -1 in examples["label"]:
                label_map = {-1: 0, 0: 1, 1: 2}
                return label_map[label]
            elif label in [0, 1, 2]:
                return label
            else:
                raise ValueError(f"Invalid label value: {label}")
        
        try:
            labels = [map_label(label) for label in examples["label"]]
            print("Mapped labels sample:", labels[:5])
        except Exception as e:
            print(f"Error mapping labels: {e}")
            raise
        
        # Tokenize texts
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None
        )
        
        # Add labels to tokenized output
        tokenized["labels"] = labels
        return tokenized

    try:
        # Print dataset info before tokenization
        print("Dataset keys:", dataset_dict["train"].column_names)
        print("Number of examples:", len(dataset_dict["train"]))
        print("Sample text:", dataset_dict["train"]["text"][0])
        print("Sample label:", dataset_dict["train"]["label"][0])
        
        # Apply tokenization and keep labels
        tokenized = dataset_dict.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset_dict["train"].column_names
        )
        
        # Set format for PyTorch
        tokenized.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        
        print(f"Tokenized dataset columns: {tokenized['train'].column_names}")
        print(f"Sample tokenized features: {tokenized['train'][0].keys()}")
        
        return tokenized
        
    except Exception as e:
        print(f"Error in tokenize_dataset: {e}")
        raise

def train_kfold(model, tokenizer, dataset, device, n_splits=5):
    """Perform k-fold cross validation"""
    try:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []
        best_f1 = 0
        best_model = None
        
        # Convert dataset to list for splitting
        dataset_dict = {'text': dataset['text'], 'label': dataset['label']}
        
        # Train each fold
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(dataset_dict['text'], dataset_dict['label'])):
            print(f"\n=== Preparing Fold {fold_num + 1}/{n_splits} ===")
            # Create fold datasets
            train_labels = [dataset_dict['label'][i] for i in train_idx]
            val_labels = [dataset_dict['label'][i] for i in val_idx]
            # Check if all classes are present in the training set
            if len(set(train_labels)) < 3:
                print(f"Skipping fold {fold_num + 1}: Not all classes present in training set ({set(train_labels)})")
                continue
            print(f"Train label counts for fold {fold_num + 1}: {np.bincount(train_labels)}")
            print(f"Validation label counts for fold {fold_num + 1}: {np.bincount(val_labels)}")
            train_fold = {
                'text': [dataset_dict['text'][i] for i in train_idx],
                'label': train_labels
            }
            val_fold = {
                'text': [dataset_dict['text'][i] for i in val_idx],
                'label': val_labels
            }
            
            # Create DatasetDict with proper format
            fold_dataset = DatasetDict({
                'train': Dataset.from_dict(train_fold),
                'validation': Dataset.from_dict(val_fold)
            })
            
            # Tokenize datasets
            tokenized_dataset = tokenize_dataset(fold_dataset, tokenizer)
            
            # Train and evaluate fold
            trainer, metrics = train_and_evaluate_fold(
                tokenized_dataset, 
                model, 
                tokenizer, 
                device, 
                fold_num, 
                n_splits
            )
            
            if trainer is not None and metrics is not None:
                fold_results.append(metrics)
                
                # Track best model
                if metrics['validation']['f1'] > best_f1:
                    best_f1 = metrics['validation']['f1']
                    # Save the best model's weights to disk
                    model.save_pretrained(f"{working_dir}/best_model_fold_{fold_num}")
                    best_model_path = f"{working_dir}/best_model_fold_{fold_num}"
                    print(f"New best model found for fold {fold_num + 1} with F1: {best_f1:.4f}")
                    print(f"Model saved to {best_model_path}")
        
        # Calculate and print average metrics
        if fold_results:
            print("\n=== Average Metrics Across Folds ===")
            avg_metrics = {
                'train_accuracy': np.mean([fold['train']['accuracy'] for fold in fold_results]),
                'train_f1': np.mean([fold['train']['f1'] for fold in fold_results]),
                'val_accuracy': np.mean([fold['validation']['accuracy'] for fold in fold_results]),
                'val_f1': np.mean([fold['validation']['f1'] for fold in fold_results])
            }
            
            print(f"Average Train Accuracy: {avg_metrics['train_accuracy']:.4f}")
            print(f"Average Train F1: {avg_metrics['train_f1']:.4f}")
            print(f"Average Validation Accuracy: {avg_metrics['val_accuracy']:.4f}")
            print(f"Average Validation F1: {avg_metrics['val_f1']:.4f}")
            
            # Reload the best model from disk
            try:
                best_model = PeftModel.from_pretrained("ProsusAI/finbert", best_model_path)
            except Exception as e:
                print(f"Error loading best model: {e}")
                best_model = None
            if best_model is not None:
                best_model.save_pretrained(f"{working_dir}/finbert_best_model")
                tokenizer.save_pretrained(f"{working_dir}/finbert_best_model")
                return best_model, avg_metrics, fold_results, best_model_path
        
        return None, None, None
        
    except Exception as e:
        print(f"Error in k-fold training: {e}")
        return None, None, None

# Check dataset format
df = pd.read_csv(traindt)
print("Columns:", df.columns.tolist())
print("\nFirst row:", df.iloc[0])
print("\nLabel distribution:", df['label'].value_counts())

# Update main function
def main():
    """Main execution function"""
    try:
        # Initialize model and tokenizer
        model_id = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            num_labels=3
        )

        # LoRA configuration
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["query", "value", "key"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_CLS
        )

        # Apply LoRA to the model
        model = get_peft_model(base_model, lora_config)
        print("LoRA modules applied.")
        print_trainable_lora_modules(model)
        
        # Set device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Load and preprocess dataset
        raw_dataset = load_dataset("csv", data_files={
            "train": traindt
        })["train"]
        
        # Validate labels
        print("\nValidating labels...")
        unique_labels = set(raw_dataset["label"])
        print(f"Unique labels in dataset: {unique_labels}")
        
        # Check if labels are in expected range
        valid_labels = {-1, 0, 1} if -1 in unique_labels else {0, 1, 2}
        invalid_labels = unique_labels - valid_labels
        if invalid_labels:
            raise ValueError(f"Invalid labels found in dataset: {invalid_labels}")
        
        # Perform k-fold training
        best_model, avg_metrics, fold_results, best_model_path = train_kfold(
            model, 
            tokenizer, 
            raw_dataset, 
            device
        )
        
        if best_model is not None:
            # Save best model
            model.save_pretrained(f"{working_dir}/finbert_best_model")  # This is correct for PEFT
            tokenizer.save_pretrained(f"{working_dir}/finbert_best_model")
            
            # Save metrics
            with open("training_metrics.txt", "w") as f:
                f.write("=== Average Metrics ===\n")
                for metric, value in avg_metrics.items():
                    f.write(f"{metric}: {value:.4f}\n")
            
            return best_model, tokenizer, avg_metrics, fold_results, best_model_path
            
        return None, None, None, None, None
        
    except Exception as e:
        print(f"Error in main: {e}")
        return None, None, None, None

if __name__ == "__main__":
    model, tokenizer, metrics, fold_results, best_model_path = main()

    # Only merge and save if best_model_path is valid
    if best_model_path:
        base_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", num_labels=3)
        best_model = PeftModel.from_pretrained(base_model, best_model_path)
        merged_model = best_model.merge_and_unload()
        merged_model.save_pretrained(f"{working_dir}/finbert_best_model_merged")

def predict_sentiment(text, model, tokenizer, device):
    """
    Predict sentiment for a given text using the trained model
    Returns: Dictionary containing prediction results including logits
    """
    try:
        # Prepare model
        model.eval()
        model = model.to(device)

        # Tokenize input
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1)[0].item()

        # Map prediction to sentiment
        sentiment_map = {0: "positive", 1: "neutral", 2: "negative"}
        confidence = probs[0][predicted_class].item()

        return {
            "text": text,
            "sentiment": sentiment_map[predicted_class],
            "confidence": f"{confidence:.4f}",
            "logits": logits[0].cpu().numpy().tolist(),
            "probabilities": {
                "positive": f"{probs[0][0].item():.4f}",
                "neutral": f"{probs[0][1].item():.4f}",
                "negative": f"{probs[0][2].item():.4f}"
            }
        }

    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def test_model(model_path=f"{working_dir}/finbert_best_model_merged"):
    """
    Test the trained model on sample texts
    """
    try:
        # Load model and tokenizer
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Sample texts for testing
        test_texts = [
            "The company reported strong earnings growth and increased dividend.",
            "The stock market remained flat amid mixed economic signals.",
            "Investors lost millions as the company filed for bankruptcy."
        ]

        # Make predictions
        results = []
        for text in test_texts:
            prediction = predict_sentiment(text, model, tokenizer, device)
            if prediction:
                results.append(prediction)
                print("\nText:", text)
                print("Sentiment:", prediction["sentiment"])
                print("Confidence:", prediction["confidence"])
                print("Logits:", prediction["logits"])
                print("Class Probabilities:", prediction["probabilities"])

        return results

    except Exception as e:
        print(f"Error in testing: {e}")
        return None


