import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
model_path = "./models/t5small-finetuned-qmsum"  # Update with your model path
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

def generate_summary_t5(input_text: str) -> str:
    """Generate a summary using a T5 model."""
    text_with_prefix = "summarize: " + input_text
    inputs = tokenizer(
        text_with_prefix,
        padding="max_length",
        truncation=True,
        max_length=512,  # Adjust based on your training settings
        return_tensors="pt"
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,  # Adjust based on desired output length
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
    
    summary = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return summary
