import argparse, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

def main(args):
    model_id = args.base_model_id
    dataset_id = args.dataset_id
    output_dir = args.output_dir

    ds = load_dataset(dataset_id)

    def to_text(ex):
        q = ex.get("question") or ex.get("input") or ex.get("prompt") or ex.get("query") or ex.get("Patient") or ex.get("user") or ""
        a = ex.get("answer") or ex.get("output") or ex.get("response") or ex.get("Doctor") or ex.get("assistant") or ""
        q, a = str(q).strip(), str(a).strip()
        return {"text": f"<s>[INST] You are a cautious, supportive medical assistant.\nUser: {q}\n[/INST]{a}"}

    for split in list(ds.keys()):
        ds[split] = ds[split].map(to_text)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        fp16=False,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds.get("train") or ds[list(ds.keys())[0]],
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=True,
        args=training_args,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA adapters to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--dataset_id", type=str, default="ruslanmv/ai-medical-chatbot")
    parser.add_argument("--output_dir", type=str, default="artifacts/lora/medical_chatbot_lora")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    args = parser.parse_args()
    main(args)
