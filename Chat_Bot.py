from datasets import load_dataset
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Trainer, TrainingArguments
import torch
import evaluate

dataset = load_dataset("daily_dialog")
train_data = dataset['train']
val_data = dataset['validation']

def create_pairs(batch):
    inputs = []
    targets = []
    for dialog in batch['dialog']:
        for i in range(len(dialog) - 1):
            inputs.append(dialog[i])
            targets.append(dialog[i+1])
    return {"input_text": inputs, "target_text": targets}

train_pairs = train_data.map(create_pairs, batched=True, batch_size=1000, remove_columns=train_data.column_names)
val_pairs = val_data.map(create_pairs, batched=True, batch_size=1000, remove_columns=val_data.column_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#  بارگذاری مدل و توکنایزر BlenderBot

model_name = "facebook/blenderbot-400M-distill"  # نسخه سبک‌تر BlenderBot 3.0
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

model.to(device)

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 64


def preprocess_function(examples):
    inputs = examples["input_text"]
    targets = examples["target_text"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
    )

    # توجه: در نسخه‌های جدید Transformers نیازی به استفاده از as_target_tokenizer نیست
    labels = tokenizer(
        targets,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_pairs.map(preprocess_function, batched=True, remove_columns=train_pairs.column_names)
tokenized_val = val_pairs.map(preprocess_function, batched=True, remove_columns=val_pairs.column_names)


# تعریف تابع ارزیابی BLEU
def evaluate_bleu(model, tokenizer, dataset, device, num_samples=100):
    bleu = evaluate.load("bleu")
    model.eval()

    predictions = []
    references = []

    for i in range(num_samples):
        example = dataset[i]

        input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(example["attention_mask"]).unsqueeze(0).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=5,
                early_stopping=True
            )

        pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        target_text = tokenizer.decode(example["labels"], skip_special_tokens=True)

        predictions.append(pred_text.split())
        references.append([target_text.split()])

    bleu_score = bleu.compute(predictions=predictions, references=references)
    return bleu_score["bleu"]


#  تنظیم Trainer و آموزش



training_args = TrainingArguments(
    output_dir="./blenderbot-dailydialog",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=100,
    fp16=False,
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,


)

trainer.train()
print("Training finished.")

#  ذخیره مدل آموزش‌دیده

trainer.save_model("./blenderbot-dailydialog-finetuned")
tokenizer.save_pretrained("./blenderbot-dailydialog-finetuned")


# ارزیابی مدل

bleu_result = evaluate_bleu(model, tokenizer, tokenized_val, device, num_samples=100)
print(f"BLEU score on validation set: {bleu_result:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# اینجا مسیر مدل ذخیره شده یا نام مدل
model_name_or_path = "./blenderbot-dailydialog-finetuned"


# بارگذاری توکنایزر و مدل از مسیر ذخیره شده (یا دانلود از اینترنت)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name_or_path)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name_or_path)
model.to(device)


def chat_with_context(model, tokenizer, device, max_input_length=128, max_output_length=64):
    model.eval()
    history = ""
    print("Start chat with model (type 'exit' to exit):")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("End Of Chat.")
            break

        # اضافه کردن ورودی کاربر به تاریخچه (کانتکست)
        history += user_input + " </s> "

        # توکنایز کردن تاریخچه (کانتکست)
        inputs = tokenizer(
            history,
            max_length=max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        # تولید پاسخ مدل
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_output_length,
                num_beams=5,
                early_stopping=True,
            )

        # دیکد کردن پاسخ مدل
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print("Chat_Bot:", response)

        # اضافه کردن پاسخ مدل به تاریخچه (کانتکست)
        history += response + " </s> "


chat_with_context(model, tokenizer, device)
