# MedMistral-Chatbot 🩺
Lightweight medical assistant chatbot fine-tuned with **QLoRA** on **Mistral-7B-Instruct** using the
`ruslanmv/ai-medical-chatbot` dataset. Includes a **Streamlit** chat UI and a **Colab-ready fine-tuning notebook**.

> ⚠️ **Disclaimer:** Educational triage guidance only; not medical advice or diagnosis.

## Quickstart
1. Open the notebook `notebooks/01_finetune_mistral_medical_QLoRA.ipynb` in **Colab/Kaggle**.
2. Run all cells → saves LoRA adapters to `artifacts/lora/medical_chatbot_lora/`.
3. Install deps and run Streamlit app:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Config
Update `config/config.yaml` to change model or generation settings.
