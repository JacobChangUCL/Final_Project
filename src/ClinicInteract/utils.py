import time
from openai import OpenAI
from cofig import Config
import requests


def evaluate(true_diagnosis, doctor_agent_diagnosis, model="gpt-4o-mini"):
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

    def __init__(self, json_object, index=None):
        self.patient_information = json_object['Main Information']
        self.disease = json_object['answer']
        self.physical_examination_findings = json_object['Physical Examination Findings']
        self.test_result = json_object['Laboratory Results']
        self.index = index

    def get_summary(self):
        """
        Return a summary of the structured data.
        """
        return {
            "Patient Information": self.patient_information,
            "Correct Diagnosis": self.disease,
            "Physical Examination Findings": self.physical_examination_findings,
            "Test Results": self.test_result,
            "Index": self.index
        }


def asking_question(model_name: str, prompt, system_prompt, max_retries=10, time_out=5):
    """
    ask the model and return the answer
    """
    valid_model = ["gpt-4o", "gpt-4o-mini", "deepseek-chat", "qwen-plus", "deepseek-reasoner", "deepseek-r1", "o3-mini"]
    if model_name not in valid_model:
        raise Exception(f"No model by the name {model_name},all the valid models are {valid_model}")

    for _ in range(max_retries):
        try:
            if model_name == "gpt-4o" or model_name == "gpt-4o-mini" or model_name == "o3-mini":
                client = OpenAI(
                    api_key=Config.openai_api_key,
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.encode('utf-8', errors='ignore').decode('utf-8')

            elif model_name == "deepseek-chat" or model_name == "deepseek-reasoner":
                # add deepseek-reasoner later

                url = "https://api.siliconflow.cn/v1/chat/completions"

                payload = {
                    "model": "deepseek-ai/DeepSeek-V3",
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                headers = {
                    "Authorization": Config.silicon_flow,
                    "Content-Type": "application/json"
                }

                response = requests.request("POST", url, json=payload, headers=headers)

                return response.json()["choices"][0]["message"]["content"].encode('utf-8', errors='ignore').decode(
                    'utf-8')

            elif model_name == "qwen-plus":
                client = OpenAI(
                    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
                    api_key=Config.ali_yun_bai_lian_api_key,
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                return response.choices[0].message.content.encode('utf-8', errors='ignore').decode('utf-8')
                # deepseek will use unicode to output something like emoji,
                # but we don't need it
        except Exception as e:
            print(e)
            time.sleep(time_out)
            continue
