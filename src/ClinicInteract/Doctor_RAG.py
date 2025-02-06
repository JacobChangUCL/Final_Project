from utils import asking_question
# 1.重新整合Doctor Agent，将其分为两个阶段
# 2.在第二阶段中加入RAG模块
# 3.将提示词和Agent文件分离
from prompt_dict import prompt_dict
from Rag_optimizer import RAG_optimizer


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

    def __init__(self, backend="gpt-4o", max_conversation=10, use_RAG=False):
        """
        Doctor doesn't need initial data in chinese clinical environment
        :param max_conversation: the max conversation between patient and doctor
        """
        self.num_conversation = 0
        self.max_conversation = max_conversation
        self.backend = backend
        self.conversation_history = ""  # history of conversation
        self.use_RAG = use_RAG

    def return_question(self, patient_answer) -> str:
        diagnostic_phrase = False  # false: asking question, true: diagnostic phase
        # 可以考虑增加历史对话的压缩机制
        """
        ask a question to the patient agent
        Jan 4th change: In the final dialogue loop, reinforce the prompt: "You must make a diagnosis.

        """
        # initial doctor question to start the conversation
        if self.num_conversation == 0:
            question = "Hello, I’m doctor. Could you tell me what’s been bothering you recently or if you have any symptoms you’d like to share?"  # let's hypnosis that the doctor will say the first sentence
            self.conversation_history += "Doctor:" + question + "\n"
            self.num_conversation += 1
            return question

        if self.num_conversation > self.max_conversation:
            return "diagnosis ended"

        # diagnostic phase
        if self.num_conversation == self.max_conversation:
            if self.use_RAG:
                rag_result = RAG_optimizer(self.conversation_history)
                rag_result = "<Reference>Below is the reference material provided for your use:" + rag_result + "End of reference material</Reference>"

            input_prompt = f"You have asked all the necessary questions. Now, you must provide the diagnostic result.Below is the conversation history: {self.conversation_history}Patient:{patient_answer}\n {rag_result if self.use_RAG else ''} Please first provide your thinking process step by step, then strictly provide the final diagnosis in the following format:  Diagnosis: [specific diagnosis]."
            print(input_prompt)
            diagnostic_phrase = True


        else:
            input_prompt = f"You can ask a maximum of {self.max_conversation} questions before making a diagnosis. So far, you have asked {self.num_conversation} questions.\nBelow is your conversation history: {self.conversation_history}\n\n{patient_answer}P\n\nNow, please continue the conversation.\nDoctor: "

        question = asking_question(self.backend, input_prompt, self.system_prompt(diagnostic_phrase))

        # we add a mechanism in there:if the doctor ready to make a diagnosis before the max_conversation,
        # we will use different system prompt to guide the doctor

        if 0 < self.num_conversation < self.max_conversation:
            if "Diagnosis:" in question:
                print("Early Stop: Below is the Initial diagnosis\n", question)
                diagnostic_phrase = True

                if self.use_RAG:
                    rag_result = RAG_optimizer(self.conversation_history)
                    rag_result = "<Reference>Below is the reference material provided for your use:" + rag_result + "End of reference material</Reference>"

                input_prompt = f"You have asked all the necessary questions. Now, you must provide the diagnostic result.Below is the conversation history: {self.conversation_history}Patient:{patient_answer}\n {rag_result if self.use_RAG else ''}   Please first provide your thinking process step by step, then strictly provide the final diagnosis in the following format:  Diagnosis: [specific diagnosis]."
                print("input_prompt", input_prompt)
                #                 input_prompt_early_stop = f"""In your last response, you stated that the current information was sufficient for a diagnosis.
                # Now, you need to carefully reconsider it; if you believe the information is clearly insufficient for confirmation, please continue asking questions using DiganosisContinue: [next question] at the end of your response."""
                # If you believe it is enough to make a diagnosis, please first provide your thinking process step-by-step,
                # then strictly provide the final diagnosis in the following format: Diagnosis: [specific diagnosis].
                # This is your reference diagnosis from your last response (which may be inaccurate and is for reference only): {question}
                # Below is the conversation history: {self.conversation_history}Patient: {patient_answer}\n
                # Please continue:"""
                #
                question = asking_question(self.backend, input_prompt, self.system_prompt(diagnostic_phrase))
        # if "DiganosisContinue:" in question:
        #     question = question.split("DiganosisContinue:")[1]
        self.conversation_history += "Patient:" + patient_answer + "\n" + "Doctor:" + question + "\n"  # 继续对话

        self.num_conversation += 1
        return question

    def system_prompt(self, is_diagnostic_phase=False) -> str:
        # we can add more templates and put them into another file
        if is_diagnostic_phase:
            core_prompt = prompt_dict["doctor_diagnosis_phase_system_prompt"]
        else:
            core_prompt = prompt_dict["doctor_questioning_phase_system_prompt"]
        return core_prompt

    def info_pressure(self):
        sys_prompt = prompt_dict["doctor_info_pressure"]

        user_prompt = f"The following is a transcript of a conversation between a doctor and a patient. Please extract the useful information and optimize the content accordingly\n{self.conversation_history}"

        pressured_conversation = asking_question("gpt-4o", sys_prompt, user_prompt)
        return pressured_conversation

    def reset(self) -> None:
        self.conversation_history = ""
        self.num_conversation = 0


if __name__ == "__main__":
    doctor = Doctor(backend="gpt-4o", max_conversation=10, use_RAG=True)
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
    doctor.num_conversation = 10  # jump to diagnostic phase
    answer = doctor.return_question("")
    print(answer)
