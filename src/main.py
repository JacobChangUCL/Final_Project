# This file defines the patient agent, doctor agent and laboratory agent,then agents will interact with each other to simulate a medical diagnosis process.
from typing import List
import openai  # OpenAI API is pretty expensive,try to find something to change it
# loading dataset
import json
from utils import asking_question, DataDistributor, evaluate
from patient import Patient
from laboratory import Laboratory
from physical_examination_findings import PhysicalExamination
from doctor import Doctor
# delete it before submitting
openai.api_key = "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"

# 逻辑：三个agent，病人、医生和实验室

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

#1.增加随机选择数据集的功能
#2.把大文件分成小文件
clinical_interact = ClinicalInteract(data)
clinical_interact.start_inference([0], total_inferences=20)
