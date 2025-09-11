import argparse
from datasets import load_dataset
from typing import Dict

def detect_fields(example: Dict):
    for qk in ["question", "prompt", "input", "query", "ask", "user", "Patient", "patient", "instruction"]:
        for ak in ["answer", "response", "output", "Doctor", "doctor", "assistant"]:
            if qk in example and ak in example:
                return qk, ak
    keys = list(example.keys())
    return (keys[0] if keys else "input"), (keys[1] if len(keys)>1 else "output")

def format_example(ex):
    qk, ak = detect_fields(ex)
    q = str(ex.get(qk, "")).strip()
    a = str(ex.get(ak, "")).strip()
    text = f"<s>[INST] You are a cautious, supportive medical assistant.\nUser: {q}\n[/INST]{a}"
    return {"text": text}

def main(args):
    ds = load_dataset(args.dataset_id)
    for split in list(ds.keys()):
        ds[split] = ds[split].map(format_example, remove_columns=ds[split].column_names)
    if args.push_to_hub:
        ds.push_to_hub(args.push_to_hub)
    else:
        print(ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", type=str, default="ruslanmv/ai-medical-chatbot")
    parser.add_argument("--push_to_hub", type=str, default=None)
    args = parser.parse_args()
    main(args)
