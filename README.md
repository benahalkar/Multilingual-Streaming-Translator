# Multilingual Streaming Translator

## Overview

The Multilingual Streaming Translator is a real-time text translation pipeline powered by Large Language Models (LLMs). This project facilitates the continuous translation of text from a source language to a target language, making it suitable for applications like live subtitling, real-time communication, and multilingual content creation.

## Key Features

*   **Real-time Translation:** Processes and translates text streams on-the-fly.
*   **Multilingual Support:** Leverages LLMs to support a wide range of language pairs.
*   **Customizable:**  Easily configurable with different LLMs and datasets.
*   **Modular Design:** Well-structured code for easy understanding and modification.

## Setup

### Prerequisites

*   Python 3.7+
*   `pip` package installer

### Installation

1.  **Clone the repository:**

    ```
    git clone https://github.com/benahalkar/Multilingual-Streaming-Translator.git
    cd Multilingual-Streaming-Translator
    ```

2.  **Install dependencies:**

    ```
    pip install -r requirements.txt
    ```

### Configuration

The `main.py` script accepts the following command-line arguments:

*   `--sourcelanguage`:  The source language code (e.g., `eng_Latn` for English).
*   `--destlanguage`: The destination language code (e.g., `fra_Latn` for French).
*   `--modelname`: The name of the pre-trained LLM to use (e.g., `facebook/nllb-200-distilled-1.3B`).  Refer to Hugging Face Model Hub for available models.
*   `--dataset`:  The path to the input text dataset.

### Usage

To run the translation pipeline, execute the `main.py` script with the desired configuration:

For eg:
```
python main.py 
--sourcelanguage eng_Latn 
--destlanguage fra_Latn 
--modelname facebook/nllb-200-distilled-1.3B 
--dataset dataset.txt
```


## Directory Tree
```
.
├── LICENSE
├── README.md
├── _model
│   ├── model
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   └── pytorch_model.bin
│   └── tokenizer
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── dataset.txt
├── dataset_files
│   ├── chicago_speech.txt
│   ├── dataset.txt
│   └── pmindia.v1.hi-en.tsv
├── generate_dataset.py
├── main.py
├── requirements.txt
└── utils
    ├── evaluate.py
    └── make_llm.py
```


## Contributing

Contributions are welcome! Please submit pull requests with detailed explanations of the changes.

## License

This project is licensed under the [MIT](https://github.com/benahalkar/Multilingual-Streaming-Translator/blob/main/LICENSE) License.

## Future Enhancements

*   Implement streaming input from audio sources.
*   Add support for more LLMs and translation APIs.
*   Improve translation quality and speed.
*   Develop a user-friendly interface.
*   Incorporate error handling and logging.

## Contact

For questions or suggestions, please contact hb2776@columbia.edu

