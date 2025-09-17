import argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

def main(args):
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )

    tok = AutoTokenizer.from_pretrained(args.base_model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=bnb_config
    )
    mdl = PeftModel.from_pretrained(mdl, args.adapter_path)

    pipe = pipeline("text-generation", model=mdl, tokenizer=tok)
    prompt = f"<s>[INST] You are a cautious, supportive medical assistant.\nUser: {args.prompt}\n[/INST]"
    out = pipe(prompt, max_new_tokens=args.max_new_tokens, temperature=0.3, top_p=0.9)[0]["generated_text"]
    print(out.split("[/INST]",1)[1].strip() if "[/INST]" in out else out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.1")
    parser.add_argument("--adapter_path", type=str, default="artifacts/lora/medical_chatbot_lora")
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--prompt", type=str, default="I have a fever and sore throat for 2 days.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    args = parser.parse_args()
    main(args)
