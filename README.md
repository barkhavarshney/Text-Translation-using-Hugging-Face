# Sanskrit-English Translation using M2M-100


#Introduction

This project aims to harness the power of Facebook's M2M-100 model for Sanskrit-English translation. The M2M-100 model is a multilingual machine translation model that supports direct translation between 100 languages without pivoting to English, making it suitable for translating Sanskrit to English.


<details>
<summary><strong>Project Setup</strong></summary>

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Hugging Face Transformers library
- Other dependencies listed in `requirements.txt`

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

</details> <details> <summary><strong>Explanation of Dependencies</strong></summary>
Torch: This is the core library of PyTorch, a deep learning framework used for model training and inference. You might need a specific version compatible with your CUDA version (e.g., cu117 for CUDA 11.7). Check PyTorch’s website for the right version based on your hardware.

Transformers: The Hugging Face library provides pre-trained models and tools for natural language processing (NLP), including the M2M-100 model used in this project.

sentencepiece: A tokenizer and text processor library that is often required by models from Hugging Face, including M2M-100.

Numpy: A fundamental package for numerical computing in Python, required by many machine learning libraries, including PyTorch.

Pandas: Useful for data manipulation and preprocessing tasks, such as loading and cleaning the dataset before model training.

scikit-learn: Provides tools for model evaluation metrics like the BLEU score, precision, recall, etc. This is optional but useful for model performance evaluation.

</details> <details> <summary><strong>Data Preparation</strong></summary>
Download the dataset: Download the Sanskrit-English parallel corpus from [source].
Data cleaning: Preprocess the data to remove any noise or unwanted characters.
Tokenization: Tokenize the dataset using the Hugging Face tokenizers.
</details> <details> <summary><strong>Model Training and Fine-Tuning</strong></summary>
Load the M2M-100 model:
    
```bash
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
```

Fine-tune the model: Fine-tune the model on the prepared dataset.
```bash
# Example code to fine-tune the model
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

</details> <details> <summary><strong>Model Evaluation and Deployment</strong></summary>
Evaluate the model using BLEU score and other metrics to determine its accuracy.
    
Deploy the model on Hugging Face by creating a new model repository and uploading the trained model files.


</details> <details> <summary><strong>Usage and Advantages</strong></summary>
Translate Sanskrit to English:
    
```bash
# Set the tokenizer to Sanskrit and English
tokenizer.src_lang = "san"
tokenizer.tgt_lang = "eng"

# Tokenize and translate
inputs = tokenizer("आपले कार्य सुरु करा", return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
```

Multilingual Capability: Supports over 100 languages without needing a pivot language.
Low-Resource Language Support: Effective for languages with limited parallel data, like Sanskrit.
State-of-the-art Performance: Leverages advanced neural machine translation techniques for high-quality translations.
