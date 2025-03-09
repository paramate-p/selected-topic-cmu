import torch
from transformers import LEDTokenizer, LEDForConditionalGeneration

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

device = get_device()
model_path = "./models/led-finetuned-qmsum"   # Update with your model path
tokenizer = LEDTokenizer.from_pretrained(model_path)
model = LEDForConditionalGeneration.from_pretrained(model_path).to(device)

def generate_summary_led(input_text: str) -> str:
    """Generate a summary using an LED model."""
    inputs_dict = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=4096,  # Adjust based on your input requirements
        return_tensors="pt"
    )
    input_ids = inputs_dict.input_ids.to(device)
    attention_mask = inputs_dict.attention_mask.to(device)
    
    # Create global attention mask for LED (set global attention on the first token)
    global_attention_mask = torch.zeros_like(attention_mask)
    global_attention_mask[:, 0] = 1
    
    with torch.no_grad():
        predicted_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            max_length=256,  # Adjust based on desired output length
            num_beams=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return summary
