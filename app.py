import os, yaml, streamlit as st, torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from utils.prompts import MISTRAL_CHAT_TEMPLATE, DEFAULT_SYSTEM_PROMPT
from utils.safety import add_disclaimer, needs_emergency_attention, EMERGENCY_MSG

st.set_page_config(page_title="MedMistral - AI Medical Assistant", page_icon="🩺", layout="wide")
st.title("🩺 MedMistral — First-Phase Medical Assistant")
st.caption("Educational triage guidance only. Not a medical diagnosis.")

with st.expander("Inference Settings", expanded=False):
    base_model_id = st.text_input("Base Model ID", value="mistralai/Mistral-7B-Instruct-v0.1")
    adapter_path = st.text_input("LoRA Adapter Path or HF Repo", value="artifacts/lora/medical_chatbot_lora")
    max_new_tokens = st.slider("Max new tokens", 64, 1024, 256, step=32)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, step=0.1)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
    use_4bit = st.checkbox("Load in 4-bit (bitsandbytes)", value=True)

cfg_path = "config/config.yaml"
if os.path.exists(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    base_model_id = cfg.get("base_model_id", base_model_id)
    adapter_path = cfg.get("adapter_path", adapter_path)

@st.cache_resource(show_spinner=True)
def load_pipeline(base_model_id: str, adapter_path: str, use_4bit: bool):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quant_config
    )
    mdl = PeftModel.from_pretrained(mdl, adapter_path)
    text_gen = pipeline("text-generation", model=mdl, tokenizer=tok)
    return text_gen, tok

with st.spinner("Loading model & adapter..."):
    pipe, tokenizer = load_pipeline(base_model_id, adapter_path, use_4bit)

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_area("Describe your symptoms or question:", height=140, placeholder="e.g., I have a fever and sore throat for 2 days...")
with col2:
    if st.button("Clear Chat"):
        st.session_state.history = []

for turn in st.session_state.history:
    st.chat_message(turn["role"]).markdown(turn["content"])

if st.button("Ask"):
    if user_input.strip():
        st.session_state.history.append({"role": "user", "content": user_input.strip()})
        st.chat_message("user").markdown(user_input.strip())

        emergency_flag = needs_emergency_attention(user_input)
        conversation = []
        for t in st.session_state.history:
            conversation.append(f"{'User' if t['role']=='user' else 'Assistant'}: {t['content']}")
        convo_text = "\n".join(conversation)
        prompt = MISTRAL_CHAT_TEMPLATE.format(system_prompt=DEFAULT_SYSTEM_PROMPT, user=convo_text)

        with st.spinner("Thinking..."):
            out = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )[0]["generated_text"]

        reply = out.split("[/INST]", 1)[1].strip() if "[/INST]" in out else out
        if emergency_flag:
            reply = f"{EMERGENCY_MSG}\n\n{reply}"
        reply = add_disclaimer(reply)
        st.session_state.history.append({"role": "assistant", "content": reply})
        st.chat_message("assistant").markdown(reply)
