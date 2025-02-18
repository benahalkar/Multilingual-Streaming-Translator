import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


def make_model(model_name):
    """
    Create a model and tokenizer from a given model name.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def save_model(model, tokenizer, folder):
    """
    Save the model and tokenizer to a specified folder.

    Args:
        model: The model to be saved.
        tokenizer: The tokenizer to be saved.
        folder (str): The path to the folder where the model and tokenizer will be saved.
    """
    model.save_pretrained(os.path.join(folder, "model"))
    tokenizer.save_pretrained(os.path.join(folder, "tokenizer"))


def load_model(folder):
    """
    Load a model and tokenizer from a specified folder.

    Args:
        folder (str): The path to the folder containing the saved model and tokenizer.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(os.path.join(folder, "model"))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(folder, "tokenizer"))
    return model, tokenizer


def create_llm(model, tokenizer, src_lang, tgt_lang):
    """
    Create a language model pipeline for translation.

    Args:
        model: The model to be used in the pipeline.
        tokenizer: The tokenizer to be used in the pipeline.
        src_lang (str): The source language code.
        tgt_lang (str): The target language code.

    Returns:
        pipeline: A translation pipeline.
    """
    model = pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
    return model


if __name__ == "__main__":
    # Define the model name
    model_name = 'facebook/nllb-200-distilled-1.3B'

    # Create the model and tokenizer
    model, tokenizer = make_model(model_name)

    # Save the model and tokenizer
    save_model(model, tokenizer, "data")

    # Load the saved model and tokenizer
    model, tokenizer = load_model("data")
