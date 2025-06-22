import os
from datasets import load_dataset
from transformers import (GPT2TokenizerFast, GPT2LMHeadModel,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


def train():
    model_name = os.getenv("HF_MODEL_NAME", "gpt2")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    tokenized_ds = dataset.map(tokenize, batched=True, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=10,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model("./trained_model")


if __name__ == "__main__":
    train()
