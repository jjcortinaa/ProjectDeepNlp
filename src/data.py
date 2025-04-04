from datasets import load_dataset
import pandas as pd
import os
from transformers import pipeline
import ast
from transformers import pipeline


import os
import pandas as pd
import ast
from datasets import load_dataset
from transformers import pipeline

def download_data(path="../data/NER_csvs", sa_model_path="../data/SA_model", sa_tokenizer_path="../data/SA_tokenizer", output_path="../data/NER_SA_csvs"):
    """
    Downloads CoNLL-2003 dataset, saves CSVs, applies sentiment analysis, and stores labeled CSVs.
    
    Args:
        path (str): Path to save NER CSVs.
        sa_model_path (str): Path to save the SA model.
        sa_tokenizer_path (str): Path to save the SA tokenizer.
        output_path (str): Path to save the final CSVs with SA labels.
    """
    os.makedirs(path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Load and save raw NER datasets
    dataset = load_dataset("conll2003", trust_remote_code=True)
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(dataset[split])
        df.to_csv(os.path.join(path, f"conll2003_{split}.csv"), index=False)

    # Load SA model and tokenizer
    sentiment_analyzer = pipeline("sentiment-analysis")
    sentiment_analyzer.model.save_pretrained(sa_model_path)
    sentiment_analyzer.tokenizer.save_pretrained(sa_tokenizer_path)

    # Reload with local model (opcional, puede omitirse si ya se tiene arriba)
    sentiment_analyzer = pipeline("sentiment-analysis", model=sa_model_path, tokenizer=sa_tokenizer_path)

    for split in ["train", "validation", "test"]:
        df = pd.read_csv(os.path.join(path, f"conll2003_{split}.csv"))
        df["sentence"] = df["tokens"].apply(ast.literal_eval).apply(lambda x: " ".join(x))
        sentences = df["sentence"].tolist()

        # Predict SA labels
        results = sentiment_analyzer(sentences)
        df["label"] = [r["label"] for r in results]

        # Save final CSV
        df.to_csv(os.path.join(output_path, f"{split}.csv"), index=False)
