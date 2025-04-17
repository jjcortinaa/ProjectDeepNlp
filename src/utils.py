import torch
from torch.jit import RecursiveScriptModule
import os
from data import NERSentimentEmbeddingDataset

def save_model(model: torch.nn.Module, name: str) -> None:
    """
    This function saves a model in the 'models' folder as a torch.jit.
    It should create the 'models' if it doesn't already exist.

    Args:
        model: pytorch model.
        name: name of the model (without the extension, e.g. name.pt).
    """

    # create folder if it does not exist
    if not os.path.isdir("models"):
        os.makedirs("models")

    # save scripted model
    model_scripted: RecursiveScriptModule = torch.jit.script(model.cpu())
    model_scripted.save(f"models/{name}.pt")

    return None

def predict_alert(sa_logits: torch.Tensor, train_loader:NERSentimentEmbeddingDataset, input_ids: torch.Tensor, ner_tags: torch.Tensor, threshold=0.3):
    """
    input_ids: [batch, maxlen(seq)]
    ner_tags: [batch, maxlen(seq)]
    sa_logits: [batch, num_sa_tags]
    """
    
    mask_pos = torch.zeros(sa_logits.shape[0])
    mask_neg = torch.zeros(sa_logits.shape[0])
    
    mask_pos[torch.abs(sa_logits[:,0]-1)<threshold] = 1.0
    mask_neg[torch.abs(sa_logits[:,1])<threshold] = 1.0

    vocab = train_loader.dataset.vocab_aux
    ner_tags_dict = {
        1: "B-PER (Begin-Person) - Inicio de una entidad de tipo persona.",
        2: "I-PER (Inside-Person) - Continuación de una entidad de tipo persona.",
        3: "B-ORG (Begin-Organization) - Inicio de una entidad de tipo organización.",
        4: "I-ORG (Inside-Organization) - Continuación de una entidad de tipo organización.",
        7: "B-MISC (Begin-Miscellaneous) - Inicio de una entidad miscelánea.",
        8: "I-MISC (Inside-Miscellaneous) - Continuación de una entidad miscelánea."
    }


    for b in range(sa_logits.shape[0]):
        tokens_text = [vocab[idx.item()] for idx in input_ids[b]]
        ner_entities = [ner_tags_dict[idx.item()] for idx in ner_tags[b] if idx.item() != 0]

        if mask_pos[b] == 1.0:
            print((b, "Positive Sentiment Alert", tokens_text, ner_entities))
        if mask_neg[b] == 1.0:
            print((b, "Negative Sentiment Alert", tokens_text, ner_entities))
    
    return 










    