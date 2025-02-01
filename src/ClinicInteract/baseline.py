# This file is designed to measure the maximum diagnostic capability of a large language model.
# Note: The diagnostic capabilities of any interactive agent should remain below this maximum threshold.
from utils import evaluate, asking_question
import json
import random
import sys

# openai-version should be 1.59.9 to support deepseek

# openai.api_key = "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"
# deepseek will use openai api format
deepseek_api_key = "sk-2723f21ed00148bf971d4d5160ca3b31"


class Dataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # remove newline characters
                line = line.strip()
                # parse the line into a Python object if the line is not empty
                self.data.append(json.loads(line))
        self.num_of_cases = len(self.data)

    def get_samples(self, sample_id_list):
        for idx in sample_id_list:
            if idx < 0 or idx >= self.num_of_cases:
                raise IndexError(f"Index {idx} is out of bounds.")
        return [[self.data[idx], idx] for idx in sample_id_list]

    def get_samples_by_number(self, num_sample: int):
        if num_sample > self.num_of_cases or num_sample <= 0:
            raise ValueError(f"Number of samples requested exceeds the total number of cases or is invalid.")
        sample_id_list_1 = random.sample(list(range(self.num_of_cases)), num_sample)
        return self.get_samples(sample_id_list_1)


class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset
        self.num_of_cases = len(dataset)
        self.system_prompt = """
            You are a professional clinical doctor answering real-world medical diagnostic problems. Below is the complete information about the patient, including key details, physical examination findings, and laboratory test results. Based on this information, reason and determine the most likely disease the patient might have. Clearly demonstrate your reasoning process step-by-step, including:
            
            List potential diseases: Based on the patient's symptoms, physical examination, and test results, initially list all possible diagnostic options.
            Eliminate diseases: Analyze each option to determine why certain diseases can be ruled out, providing specific reasons. If new possibilities emerge during the analysis, add them to the list of options.
            Select the most probable disease: After ruling out incompatible options, select the disease that best aligns with the patient's information.
            State the diagnostic conclusion: Once your reasoning is complete, clearly present your final diagnosis using the following format:
            [Diagnosis: xxxdisease].
            """

    def start_evaluation(self, model_name="gpt-4o"):
        correct_num = 0
        for idx, _case in enumerate(self.dataset, start=1):
            print(f"Case {_case[1] + 1}")
            _case_copy = _case[0].copy()
            _case_copy.pop("answer")

            prompt = f"Below is all the information about the patient.{_case_copy}"

            answer = asking_question(model_name, prompt, self.system_prompt)
            print(answer)
            doctor_diagnosis = answer.split("Diagnosis:")[1]
            print(doctor_diagnosis)
            print(f"True disease is {_case[0]['answer']},doctor diagnosis is {doctor_diagnosis}")
            flag = evaluate(_case[0]["answer"], doctor_diagnosis)  # "Yes" or "No"
            if flag == "Yes":
                correct_num += 1
            print(f"Accuracy={correct_num / idx:.2%},total {idx}cases,{correct_num}are right")
        print(f"Finished.Total accuracy: {correct_num / self.num_of_cases}")
        return correct_num / self.num_of_cases


# loading dataset
file_path = "../datasets/filtered_medqa_test_set_final_version.jsonl"
# add random selection of dataset
save_path = "../experiment_record/deepseek-num50_fullInformation"

# with open(save_path, "w", encoding="utf-8") as file:
#     sys.stdout = file
num_cases = 50
dataset = Dataset(file_path).get_samples_by_number(num_cases)
evaluator = Evaluator(dataset)
evaluator.start_evaluation("deepseek-chat")
