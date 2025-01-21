import openai
import time


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
