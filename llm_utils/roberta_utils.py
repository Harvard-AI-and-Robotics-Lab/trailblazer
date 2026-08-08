from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

ckpt_path = 'hubert233/GPTFuzz'
model = None
tokenizer = None


def load_roberta(device=None):
    global model, tokenizer
    if model is None or tokenizer is None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Loading RoBERTa Checkpoint...")
        model = RobertaForSequenceClassification.from_pretrained(ckpt_path).to(device)
        tokenizer = RobertaTokenizer.from_pretrained(ckpt_path)
        print("Loading Done!")
    return model, tokenizer


def predict(sequences, device=None):
    model_, tokenizer_ = load_roberta(device=device)
    model_device = next(model_.parameters()).device
    # Encoding sequences
    inputs = tokenizer_(sequences, padding=True, truncation=True, max_length=512, return_tensors="pt").to(model_device)

    # Compute token embeddings
    with torch.no_grad():
        outputs = model_(**inputs)

    # Get predictions
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # If you want the most likely classes:
    _, predicted_classes = torch.max(predictions, dim=1)


    return predicted_classes
