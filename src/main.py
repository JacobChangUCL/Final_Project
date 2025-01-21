# This file defines the patient agent, doctor agent and laboratory agent,then agents will interact with each other to simulate a medical diagnosis process.
from typing import List
import openai  # OpenAI API is pretty expensive,try to find something to change it
import time
# loading dataset
import json
from tqdm import tqdm
import time

# delete it before submitting
openai.api_key = "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"


class DataDistributor:
    """
    Structure data from a json object.
    """

    def __init__(self, json_object):
        self.patient_information = json_object['Main Information']
        self.disease = json_object['answer']
        self.physical_examination_findings = json_object['Physical Examination Findings']
        self.test_result = json_object['Laboratory Results']

    def get_summary(self):
        """
        Return a summary of the structured data.
        """
        return {
            "Patient Information": self.patient_information,
            "Correct Diagnosis": self.disease,
            "Physical Examination Findings": self.physical_examination_findings,
            "Test Results": self.test_result,
        }


# 逻辑：三个agent，病人、医生和实验室
# 实验室是个函数，接受医生的测试请求，返回测试结果。如果没有相关的测试则返回正常。

class Laboratory:
    """
    Return measurement results based on the doctor's request.
    """
    prompt = "You will act as a medical laboratory responsible for responding to doctors' test requests."

    def __init__(self, data_distributor):
        self.Test_Results = data_distributor.test_result

    def get_result(self, doctor_request):
        """
        Return the result of the test.
        """
        messages = [
            {"role": "system", "content": Laboratory.prompt},

            {"role": "user", "content": f"The doctor requested a test: {doctor_request}\n"
                                        f"Here is the list of test results: {self.Test_Results}\nPlease strictly respond with the corresponding test result from the list. "
                                        f"Please remember that sometimes the tests requested by the doctor include multiple test results, and you need to output all the included results. For example, "
                                        f"for 'Complete Blood Count (CBC)', return items such as 'Hemoglobin (Hb)', 'RBC Count', 'Hematocrit (HCT)', Mean Corpuscular Volume (MCV)', 'Mean Corpuscular Hemoglobin (MCH)', and 'Mean Corpuscular Hemoglobin Concentration (MCHC)'."
                                        f"If the test is not found in the results, reply with 'Result is normal.' Do not add any additional information."}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
        )

        return response.choices[0].message['content']


def asking_question(model_name, prompt, system_prompt, max_retries=2, time_out=20):
    """
    ask the model and return the answer
    """
    valid_model = ["gpt4o"]
    if model_name not in valid_model:
        raise Exception(f"No model by the name {model_name},all the valid models are {valid_model}")

    for _ in range(max_retries):
        try:
            if model_name == "gpt4o":
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response["choices"][0]["message"]["content"]


        except Exception as e:
            print(e)
            time.sleep(time_out)
            continue


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

    def system_prompt(self, bias=None) -> str:

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


def evaluate(true_diagnosis, doctor_agent_diagnosis, model="gpt4o"):
    """
    Evaluate whether the diagnosis from the doctor agent matches the true diagnosis.

    Args:
        true_diagnosis (str): The correct diagnosis.
        doctor_agent_diagnosis (str): The diagnosis provided by the doctor agent.
        model (str): The name of the model to be used for evaluation. Default is "gpt4o".

    Returns:
        str: "yes" if the diagnoses match, "no" otherwise.
    """
    system_prompt = (
        "You are a professional evaluator for medical diagnostics. "
        "Your task is to compare a correct medical diagnosis with a doctor's diagnosis. "
        "Determine if they represent the same disease. "
        "Your response must strictly be either 'Yes' or 'No'—nothing else. "
        "Maintain objectivity and base your judgment solely on the provided information."
    )

    user_prompt = (
        f"Here is the correct diagnosis: {true_diagnosis}\n"
        f"Here was the doctor's diagnosis: {doctor_agent_diagnosis}\n"
        "Are these the same disease? Please respond only with 'Yes' or 'No'."
    )
    answer = asking_question(model_name=model, prompt=user_prompt, system_prompt=system_prompt)
    return answer


class PhysicalExamination:
    def __init__(self, data_distributor):
        self.exam_result = data_distributor.physical_examination_findings

    def get_result(self):
        return self.exam_result


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
        >>> # 创建一个Doctor对象
        >>> doctor = Doctor(backend="gpt4o", max_conversation=5)
        >>> # 医生向病人提问
        >>> question = doctor.return_question("我最近胸口有点闷，偶尔伴随咳嗽。")
        >>> print(question)
        # 输出一段新的问题或检查请求，例如“你咳嗽有痰吗？持续了多久？”
    """

    def __init__(self, backend="gpt4o", max_conversation=10):
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

        print("num_conversation=", self.num_conversation)
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
        4. After gathering enough information, provide the final diagnosis using the format: Diagnosis: [Specific Diagnosis].
        5. Once you make a diagnosis in the historical response, you must immediately stop the conversation and only reply with "Diagnosis: [Specific Diagnosis]."
        6. Minimize the total number of questions while gathering the necessary information.

        Please follow these guidelines:
        - Avoid repeating questions that have already been asked. If no new questions are necessary, make the diagnosis as soon as possible.
        - Each conversation should be no more than 3 sentences, maintaining professionalism and conciseness.
        - Only request laboratory tests when required.
        - If the patient asks irrelevant or temporary questions, provide a brief response or re-emphasize the focus.
        - Once you feel enough information has been gathered, make the diagnosis promptly to avoid excessive questioning.
        Please begin your consultation based on these rules and adjust your strategy according to the patient's responses.
        """
        return core_prompt

    def reset(self) -> None:
        self.conversation_history = ""
        self.num_conversation = 0


class ClinicalInteract:
    """
    A class to simulate interactions between patients, doctors, and laboratories.
    This class handles retrieving patient data samples and initiating the inference
    process for the given patient cases.

    Attributes:
        data (List[dict]): The dataset containing patient case information.
        num_of_cases (int): The total number of patient cases in the dataset.
    """

    def __init__(self, data):
        """
        Initializes the ClinicalInteract object with the provided dataset.

        Args:
            data (List[dict]): A list of dictionaries representing patient case data.
        """
        self.data = data
        self.num_of_cases = len(data)

    def get_samples(self, index_list: List[int]) -> List[DataDistributor]:
        """
        Retrieves samples from the dataset based on the provided list of indices.
        Ensures all indices are valid before returning the corresponding samples.

        Args:
            index_list (List[int]): A list of indices representing the samples to retrieve.

        Returns:
            List[DataDistributor]: A list of DataDistributor objects corresponding to the samples.

        Raises:
            IndexError: If any index in the index_list is out of bounds of the available cases.
        """
        # boundary check
        for idx in index_list:
            if idx < 0 or idx >= self.num_of_cases:
                raise IndexError(f"Index {idx} is out of bounds.")
        return [DataDistributor(self.data[idx]) for idx in index_list]

    def start_inference(self, sample_id_list: List[int], total_inferences=5):
        """
        Starts the inference process for the given list of sample IDs.

        Args:
            sample_id_list (List[int]): A list of sample IDs to be used for inference.
            total_inferences (int, optional): The number of inferences to perform. Defaults to 10.

        """
        samples = self.get_samples(sample_id_list)

        correct_num = 0  # correct number of total cases

        for idx, _case in enumerate(samples, start=1):
            # initial data
            patient_agent = Patient(_case)
            # 缺医生助理
            physicalExamination = PhysicalExamination(_case)
            laboratory = Laboratory(_case)
            doctor = Doctor(max_conversation=total_inferences)

            # start inference
            patient_answer = ""
            early_stop = False  # if the diagnosis ended before the total_inferences,
            # this flag will be set to True
            for round_idx in range(total_inferences):  # inference stage+1 decision making stage
                print(f"Round {round_idx + 1}")
                doctor_question = doctor.return_question(patient_answer)  # patient return answer
                print("Doctor:" + doctor_question)

                if "Order test" in doctor_question:
                    test_name = doctor_question.split(":")[1:]
                    print(f"Test Name:{test_name}")
                    laboratory_result = laboratory.get_result(test_name)
                    print(f"Laboratory:{laboratory_result}")
                    # If doctor ask for lab test,the Lab agent will replace the patient agent to answer the question
                    patient_answer = "Laboratory Result:" + laboratory_result
                    continue

                if "Order Physical Examination" in doctor_question:
                    physicalExamination_result = physicalExamination.get_result()
                    print(f"Physical  Examination Result:{physicalExamination_result}")
                    patient_answer = "Physical  Examination Result:" + physicalExamination_result
                    continue
                if "Diagnosis:" in doctor_question:
                    print("Diagnosis ended")
                    early_stop = 1
                    break
                patient_answer = patient_agent.return_question(doctor_question)  # doctor ask question
                print("Patient:" + patient_answer)

            if early_stop:
                doctor_diagnosis = doctor_question
            else:
                doctor_diagnosis = doctor.return_question(patient_answer)

            print(f"True disease is {_case.disease},doctor diagnosis is {doctor_diagnosis}")
            if evaluate(_case.disease, doctor_diagnosis) == "Yes":
                correct_num += 1
            print(f"Accuracy={correct_num / idx:.2%},total {idx}cases,{correct_num} are right")


# Now we can test the doctor agent
file_path = "datasets/filtered_medqa_test_set_final_version.jsonl"
data = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        # 去掉换行符
        line = line.strip()
        # 如果这行内容不是空的，就解析成 Python 对
        data.append(json.loads(line))

clinical_interact = ClinicalInteract(data)
clinical_interact.start_inference([0], total_inferences=20)
