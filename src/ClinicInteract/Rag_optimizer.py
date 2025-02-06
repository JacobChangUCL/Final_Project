from utils import asking_question
from src.RAG import rag



def summarize_dialog(dialog_text):
    prompt = f""" Please summarize the following conversation, 
    extracting all key information, especially the patient's condition description, symptoms, 
    signs, medical history, and the doctor's potential diagnoses.\n\n
            {dialog_text}\n\n
            Please output only one summary."""
    system_prompt = "You are a summarization assistant."

    response = asking_question("deepseek-chat", prompt, system_prompt)

    key_word_for_distinguish = "diagnosis  diagnpsis distinguish  rule out diagnostic criteria distinguishing features"
    return response + key_word_for_distinguish


def RAG_optimizer(input_text):
    query = summarize_dialog(input_text)
    value = rag.answer(query)
    return value[0][0]

if __name__ == "__main__":
    dialog_history = """Round 1
Doctor:Hello, I’m doctor. Could you tell me what’s been bothering you recently or if you have any symptoms you’d like to share?
Patient:I've been experiencing swelling and pain in my right knee, and my left knee was also swollen a few weeks ago. It's really making it hard for me to walk, and I'm feeling quite frustrated. Additionally, I've been dealing with some back pain that makes it tough to move around during the day.
Round 2
Doctor:Have you experienced any fever, fatigue, or unexplained weight loss alongside these symptoms? Also, do you have a history of joint or back conditions?
Patient:No, I haven't experienced any fever, fatigue, or unexplained weight loss. I do have a history of arthritis, which I was diagnosed with 16 years ago.
Round 3
Doctor:What type of arthritis were you diagnosed with, and have you been taking any medications for it? Additionally, has the swelling or pain in your knees improved or worsened over time?
Patient:I was diagnosed with osteoarthritis. I've been managing it with some over-the-counter pain relievers. The swelling and pain in my knees have worsened recently, especially with both joints now affected.
Round 4
Doctor:Have you noticed any redness or warmth around the swollen areas of your knees? Also, has there been any recent injury or excessive strain on your joints?
Patient:No, I haven't noticed any redness or warmth around my knees. There hasn't been any recent injury or excessive strain; it just seems to be worsening on its own.
Round 5
Doctor:Given your history of osteoarthritis and the worsening symptoms without signs of infection or injury, I’ll order an X-ray to assess the joint damage. Order test: Knee X-ray.
Test Name:[' Knee X-ray.']
Laboratory:Result is normal.
Round 6
Doctor:Given the normal X-ray results and your history of osteoarthritis, I’ll order a physical examination to further assess the swelling and pain. Order Physical Examination.
Physical  Examination Result:Pitting of his nails
"""
    print(RAG_optimizer(dialog_history))
