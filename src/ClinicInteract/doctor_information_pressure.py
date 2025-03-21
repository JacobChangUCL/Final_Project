from utils import asking_question


class Doctor:
    """
    Doctor类用于模拟医生与病人之间的对话、提问和诊断流程。其功能包括：
    1. 根据病人回复（patient_answer）生成后续问题或检查请求；
    2. 记录并管理对话历史；(self.conversation_history)
    3. 在达到设定的最大提问次数后，结束对话并进行诊断。

    Attributes:
        num_conversation (int): 已向病人提出的问题数量。
        max_conversation (int): 医生可提出的最大问题数量，在此之后需结束对话。
        backend (str): 用于生成对话问题的语言模型名称(default:GPT4o)。
        conversation_history (str): 医患对话的历史记录。

    Methods:
        __init__(backend="gpt4o", max_converstion=20):
            初始化Doctor实例，并设置对话后端与最大提问次数。

        return_question(patient_answer: str) -> str:
            根据病人的回答生成新的提问或检查请求，若已超出最大提问次数则返回“diagnosis ended”。

        system_prompt() -> str:
            生成对患者端的系统提示语，用于指导患者如何回答医生的问题。

        reset() -> None:
            重置对话记录和问题计数器，以便进行新的诊断会话。

    Usage Example:
        # 创建一个Doctor对象
        doctor = Doctor(backend="gpt4o", max_conversation=5)
         # 医生向病人提问
        question = doctor.return_question("我最近胸口有点闷，偶尔伴随咳嗽。")
     print(question)
        # 输出一段新的问题或检查请求，例如“你咳嗽有痰吗？持续了多久？”
    """

    def __init__(self, backend="gpt-4o", max_conversation=10):
        """
        Doctor doesn't need initial data in chinese clinical environment
        :param max_conversation: the max conversation between patient and doctor
        """
        self.num_conversation = 0
        self.max_conversation = max_conversation
        self.backend = backend
        self.conversation_history = ""  # history of conversation

    def return_question(self, patient_answer) -> str:
        # 可以考虑增加历史对话的压缩机制
        """
        ask a question to the patient agent
        Jan 4th change: In the final dialogue loop, reinforce the prompt: "You must make a diagnosis.

        """
        if self.num_conversation == 0:
            question = "Hello, I’m doctor. Could you tell me what’s been bothering you recently or if you have any symptoms you’d like to share?"  # let's hypnosis that the doctor will say the first sentence
            self.conversation_history += "Doctor:" + question + "\n"
            self.num_conversation += 1
            return question

        if self.num_conversation > self.max_conversation:
            return "diagnosis ended"

        if self.num_conversation == self.max_conversation:
            input_prompt = f"You have asked all the necessary questions. Now, you must provide the diagnostic result.Below is the conversation history: {self.conversation_history}Patient:{patient_answer}\nPlease provide the final diagnosis in the following format: Diagnosis: [specific diagnosis]."


        else:
            input_prompt = f"You can ask a maximum of {self.max_conversation} questions before making a diagnosis. So far, you have asked {self.num_conversation} questions.\nBelow is your conversation history: {self.conversation_history}\n\n{patient_answer}P\n\nNow, please continue the conversation.\nDoctor: "

        question = asking_question(self.backend, input_prompt, self.system_prompt())

        self.conversation_history += "Patient:" + patient_answer + "\n" + "Doctor:" + question + "\n"  # 继续对话

        self.num_conversation += 1
        return question

    def system_prompt(self) -> str:
        # we can add more templates and put them into another file
        core_prompt = """You are a professional clinical doctor conducting a conversation with a patient. You can ask questions to gather the patient's medical history, symptoms, and other crucial information, or you may request laboratory tests. The laboratory will provide corresponding test results based on your requests. If the results are normal and not critical, they will return "Normal result.You can also request physical examination findings from the 'Doctor Assistant' using the command: Order Physical Examination
        Your objectives are:
        1. Collect sufficient medical information from the patient through a limited number of questions (e.g., a maximum of 10 questions).
        2. Request laboratory tests when necessary, following this strict format: Order test: [Test Name].
        3. You may also request physical examinations through an assistant, strictly following this format: Order Physical Examination.
        4. You can only request either a physical examination or laboratory tests in a single conversation. If you need both tests, please put them in two separate conversations.
        5. After gathering enough information, provide the final diagnosis using the format: Diagnosis: [Specific Diagnosis].
        6. Once you make a diagnosis in the historical response, you must immediately stop the conversation and only reply with "Diagnosis: [Specific Diagnosis]."
        7. Minimize the total number of questions while gathering the necessary information.

        Please follow these guidelines:
        - Avoid repeating questions that have already been asked. If no new questions are necessary, make the diagnosis as soon as possible.
        - Each conversation should be no more than 3 sentences, maintaining professionalism and conciseness.
        - Only request laboratory tests when required.
        - If the patient asks irrelevant or temporary questions, provide a brief response or re-emphasize the focus.
        - Once you feel enough information has been gathered, make the diagnosis promptly to avoid excessive questioning.
        Please begin your consultation based on these rules and adjust your strategy according to the patient's responses.
        """
        return core_prompt

    def info_pressure(self):
        sys_prompt = """You are an efficient and accurate conversation extraction assistant.
Your task is to extract useful information from interactions between doctors and patients, as well as between doctors and laboratory or physical examination systems, and convert it into declarative sentences while filtering out irrelevant information.
Please strictly follow the rules below:

1. Output Format
Each segment of the conversation should be described as a single declarative sentence, without retaining the original question.
Each declarative sentence should be preceded by a numerical index for clarity.
Use the original wording of the patient and doctor as much as possible, unless the original text is too lengthy or repetitive.
Preserve as much original information as possible.
2. Information Filtering
Remove the following:

Meaningless or repetitive expressions (e.g., "um," "oh," "thank you").
Small talk or personal comments unrelated to diagnosis.
Other content unrelated to medical diagnosis and treatment.
(However, note that certain facts, such as travel history, may contain diagnostic-relevant information and should be retained.)
3. Content Optimization
Prioritize extracting the following information:

Symptoms: e.g., headache, nausea, fever, etc.
Medical history: e.g., chronic diseases, past surgeries, drug allergies, etc.
Test results: e.g., laboratory test data, imaging results, physical examination findings.
Patient information: e.g., personal details, travel history, occupational history, family medical history, daily lifestyle.
Diagnostic recommendations: e.g., the doctor’s preliminary judgment, suggested tests, treatment plans.
4. Example
Input Conversation
Doctor: What discomfort or symptoms have you been experiencing today?
Patient: I have a headache, along with nausea, which started last night.
Doctor: Have you measured your temperature?
Patient: Yes, I have. I did not have a fever.
Doctor: Please check the patient's blood pressure.
Physical Examination Result: Blood pressure is 130/90 mmHg.

Output
The patient has had a headache and nausea since last night.
The patient has measured their temperature, and the result showed no fever.
The physical examination showed a blood pressure of 130/90 mmHg."""

        user_prompt = f"The following is a transcript of a conversation between a doctor and a patient. Please extract the useful information and optimize the content accordingly\n{self.conversation_history}"

        pressured_conversation = asking_question("gpt-4o", sys_prompt, user_prompt)
        return pressured_conversation

    def reset(self) -> None:
        self.conversation_history = ""
        self.num_conversation = 0

if __name__ == "__main__":
    doctor = Doctor(backend="gpt4o", max_conversation=10)
    doctor.conversation_history = """Doctor: Hello, I’m doctor. Could you tell me what’s been bothering you recently or if you have any symptoms you’d like to share?
Patient: Hi, doctor. Recently, I've been experiencing swelling and pain in my right knee. My left knee was swollen a few weeks ago, but now it's affecting both, making it hard to walk. I also have back pain that's been making it really difficult to get around during the day. The pain does get better when I rest.

Doctor: Have you had any recent injuries or experienced any fever, weight loss, or fatigue?
Patient: No, I haven't had any recent injuries, fever, or weight loss. I do feel fatigued due to the pain and difficulty moving around, though.

Doctor: Do you have any other joint pain, stiffness, or swelling elsewhere in your body, and have you noticed any changes in your skin, such as rashes?
Patient: I don't have any other joint pain or stiffness apart from my knees and back. I haven't noticed any rashes, but I do have dandruff for which I use special shampoos.

Doctor: Have you ever been diagnosed with a chronic condition like arthritis or psoriasis, or do you have a family history of such conditions?
Patient: Yes, I was diagnosed with arthritis 16 years ago. There is no family history of arthritis or psoriasis that I know of.

Doctor: Have you noticed any recent increase in the discomfort or stiffness in the morning, and do the symptoms improve with activity?
Patient: Yes, I've noticed more stiffness in the morning recently, but the symptoms seem to improve with rest rather than activity.

Doctor: Thank you for the information. Have you recently started any new medications or supplements, or stopped any, that might be related to these symptoms?
Patient: No, I haven't started or stopped any new medications or supplements recently.

Doctor: Have you experienced any recent changes in your overall health, such as urinary problems or new bowel symptoms?
Patient: No, I haven't experienced any recent changes in my overall health, like urinary problems or new bowel symptoms.

Doctor: Can you describe the nature of your back pain? Is it more of a sharp, shooting pain, or a dull ache? Additionally, have you been experiencing any eye problems such as dryness, redness, or irritation?
Patient: My back pain is more of a dull ache that makes it hard to move around. As for my eyes, I haven't experienced any dryness, redness, or irritation recently.

Doctor: Thank you for the detailed information. To assist with the diagnosis, let's conduct a physical examination for more insights. Order Physical Examination.
Physical Examination Result: Pitting of his nails."""
    squeezed_conversation = doctor.info_pressure()
    print(squeezed_conversation)
