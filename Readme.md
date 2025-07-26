README.md
markdown
Copy
Edit
# BlenderBot Fine-tuning on DailyDialog Dataset

This project fine-tunes the BlenderBot model (the lighter version `facebook/blenderbot-400M-distill`) on the DailyDialog dataset. After training, the model can be used for interactive chatting with context tracking.

---

## Features

- Load and preprocess the DailyDialog dataset
- Convert dialogs into input-target pairs for training
- Use pretrained BlenderBot 400M Distill model
- Fine-tune the model using Hugging Face Transformers and Trainer API
- Evaluate the model with BLEU score metric
- Interactive chat interface with context preservation

---

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt