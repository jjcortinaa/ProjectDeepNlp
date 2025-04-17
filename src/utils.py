import torch
from torch.jit import RecursiveScriptModule
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # type: ignore
from typing import Any, List, Dict

# Use FLAN-T5 for better quality and instruction-following capabilities
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def save_model(model: torch.nn.Module, name: str) -> None:
    """
    Save a model in the 'models' folder as TorchScript.
    Parameters:
    - model: The model to be saved (PyTorch module)
    - name: The name of the model file to save
    This function saves the model in TorchScript format for optimized deployment.
    """
    if not os.path.isdir("models"):
        os.makedirs("models")
    # Scripting the model into TorchScript format
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")


def predict_alert(
    sa_logits: torch.Tensor,
    train_loader: Any,
    input_ids: torch.Tensor,
    ner_tags: torch.Tensor,
    threshold: float = 0.2,
) -> List[str]:
    """
    Generates structured alerts using a generative model based on sentiment and NER.
    Parameters:
    - sa_logits: Sentiment analysis logits, used to determine the sentiment (positive/negative).
    - train_loader: DataLoader containing the training dataset (for accessing vocabulary).
    - input_ids: Token IDs corresponding to the words in the input text.
    - ner_tags: Named entity recognition (NER) tags.
    - threshold: Threshold for determining sentiment (default is 0.2).
    Returns:
    - A list of generated alerts in string format.
    This function creates structured alerts using sentiment and NER information,
    filtered by sentiment score and NER entities.
    """
    alerts = []

    # Apply stricter conditions to avoid false positives
    mask_pos = sa_logits[:, 0] > threshold
    mask_neg = sa_logits[:, 1] > threshold

    # Get vocabulary from the training dataset
    vocab = train_loader.dataset.vocab_aux
    ner_tags_dict: Dict[int, str] = {
        0: "Outside of a named entity",
        1: "Beginning of a person entity",
        2: "Inside of a person entity",
        3: "Beginning of an organization entity",
        4: "Inside of an organization entity",
        5: "Beginning of a location entity",
        6: "Inside of a location entity",
        7: "Beginning of a miscellaneous entity",
        8: "Inside of a miscellaneous entity",
    }

    # Loop through each batch and generate alerts
    for b in range(sa_logits.shape[0]):
        # Extract tokens and entities for each example in the batch
        tokens = [vocab[idx.item()] for idx in input_ids[b] if idx.item() in vocab]
        entities = [
            ner_tags_dict[idx.item()]
            for idx in ner_tags[b]
            if idx.item() in ner_tags_dict and idx.item() != 0
        ]
        sentiment = "positive" if mask_pos[b] else "negative" if mask_neg[b] else None

        if sentiment:
            # Generate the alert if sentiment is detected
            output = generate_alert_flan(sentiment, entities, tokens)
            if output:
                alerts.append(f"[{sentiment.upper()} ALERT] {output}")

    return alerts


def generate_alert_flan(
    sentiment: str, ner_entities: List[str], words: List[str], max_length: int = 50
) -> str:
    """
    Uses FLAN-T5 to generate a structured English alert based on sentiment and named entities.
    Parameters:
    - sentiment: The sentiment of the text ("positive" or "negative").
    - ner_entities: List of named entities recognized in the text.
    - words: List of words (tokens) from the input text.
    - max_length: Maximum length of the generated output (default is 50 tokens).
    Returns:
    - A structured alert in string format, or an empty string if the generation fails.
    This function generates an alert using a structured prompt and filters out bad or redundant outputs.
    """
    if not words or sentiment not in {"positive", "negative"}:
        return ""

    # Create a string for the named entities and words
    entities_str = ", ".join(ner_entities) if ner_entities else "no entities"
    text_str = " ".join(words)

    # Structured prompt for FLAN-T5 to follow
    prompt = (
        f"Given this information, generate alerts with the context given:"
        f"Sentiment: {sentiment}"
        f"Recognized Entities: {entities_str}"
        f"Text Content: {text_str}"
    )

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # Generate an alert using FLAN-T5 with beam search
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )

    # Decode the output text
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Filters to avoid undesirable outputs
    if not decoded or len(decoded.split()) < 5:
        return ""
    if "describe the sentiment" in decoded.lower():
        return ""
    if decoded.lower().startswith("'alert") or decoded.lower().startswith("output:"):
        return ""
    if decoded.lower() in {"none", "text", "alert", ";"}:
        return ""

    return decoded
