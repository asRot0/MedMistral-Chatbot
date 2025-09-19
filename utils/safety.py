RED_FLAG_KEYWORDS = [
    "severe chest pain", "chest pain", "shortness of breath", "difficulty breathing",
    "unable to breathe", "blue lips", "stroke", "face drooping", "slurred speech",
    "weakness on one side", "seizure", "heavy bleeding", "uncontrolled bleeding",
    "suicidal", "suicide", "homicidal", "overdose", "allergic reaction", "anaphylaxis",
    "fainting", "loss of consciousness", "severe headache", "worst headache of my life",
    "neck stiffness", "high fever", "stiff neck", "confusion", "poisoning",
]

DISCLAIMER = (
    "⚠️ **Important:** I am not a medical professional. "
    "This is general information for first-phase guidance only. "
    "Always consult a licensed clinician for diagnosis and treatment."
)

EMERGENCY_MSG = (
    "🚨 Your symptoms may indicate a medical emergency. "
    "Please seek immediate care (e.g., call your local emergency number or go to the nearest ER)."
)

def needs_emergency_attention(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in RED_FLAG_KEYWORDS)

def add_disclaimer(response: str) -> str:
    return f"{DISCLAIMER}\n\n{response}"
