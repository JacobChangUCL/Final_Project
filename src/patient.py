from utils import asking_question


class Patient:
    """
        A class representing a patient in a conversation with a doctor, including their biases and information.
        It generates responses based on the patient's medical history and bias type.
    """

    def __init__(self, data_distributor, backend="gpt4o", bias=None):

        self.info = data_distributor.patient_information
        self.backend = backend
        self.conversation_history = ""  # history of conversation
        self.bias_dict = {
            "distrust": "\nYou may have a lack of trust in AI or the healthcare system in general, which affects how you interact with your doctor or medical advisor.\n",
            "preconceived_diagnosis": "\nYou may have already diagnosed yourself, heard from a friend, or read online about a certain illness. This prior belief may influence how you approach the doctor or medical agent and could impact the conversation.\n",
            "non_scientific": "\nYou may hold certain religious or non-scientific beliefs that influence your understanding of your symptoms, leading to incorrect conclusions about the cause of your illness.\n",
            "false_memory": "\nYou might be misremembering some details about your symptoms or their timeline, which can affect the accuracy of your medical history and diagnosis.\n",
            "poor_communication": "\nYou might have difficulty clearly expressing your symptoms due to language barriers, an accent, or difficulties in organizing your thoughts. This could affect how you communicate with your doctor.\n",
            "emotional_bias": "\nYour anxiety, fear, or anger may cause you to exaggerate or downplay certain symptoms, which may impact how you communicate with the doctor.\n",
            "information_overload": "\nYou might be providing too much irrelevant or repetitive information, which can make it difficult for your doctor to focus on the key symptoms or concerns.\n",
            "self_presentation": "\nYou may be withholding certain symptoms or information due to privacy concerns or a desire to protect your self-image. This might affect the accuracy of your medical history.\n",
            "economic_pressure": "\nYou may be reluctant to undergo certain tests or treatments due to financial constraints or distrust of the healthcare system, which can affect your interaction with the doctor.\n"
        }
        if bias is not None:
            if bias not in self.bias_dict:
                raise ValueError(f"Invalid bias type. Choose from: {list[self.bias_dict.keys()]}")
            self.bias = bias
            self.bias_prompt = self.bias_dict[self.bias]
        else:
            self.bias = None
            self.bias_prompt = ""

    def return_question(self, doctor_question) -> str:
        # 可以考虑增加历史对话的压缩机制
        """
        Return the answer of doctor's question.
        """
        input_prompt = f"Below is all of your information. {self.info}Your conversation history is as follows: " + self.conversation_history + "\nThe doctor's reply is: " + doctor_question + "\nPlease proceed with the conversation\nPatient: "
        answer = asking_question(self.backend, input_prompt, self.system_prompt())
        self.conversation_history += doctor_question + "\n\n" + answer + "\n\n"  # 继续对话
        return answer

    def system_prompt(self) -> str:

        # we can add more templates and put them into another file
        core_prompt = """You are role-playing as a patient in a clinic. A doctor is inspecting you to identify your disease by asking questions and performing examinations. Your task is to respond naturally and concisely to the doctor's questions.
            Instructions:
            1. Strictly answer the questions based on the "Patient Information." If the answer to the doctor's question is found in the "Patient Information," respond according to its content. If it is not found in the "Patient Information," respond the doctor with normal information."
            2. Only answer in dialogue form, as if you were speaking directly to the doctor.
            2. Your responses should be brief, realistic, and limited below 4 sentences.
            3. Provide information that a patient might reasonably say in this scenario.
            4. Do not elaborate or provide excessive details unless explicitly asked.
            5. When answering the doctor's questions, you may recall some information that is highly relevant to your condition, but you should not bring up content that is less related."""

        patient_info_prompt = f"\n\nBelow is all of your information. {self.info}."
        return core_prompt + self.bias_prompt + patient_info_prompt

    def reset(self) -> None:
        self.conversation_history = ""
