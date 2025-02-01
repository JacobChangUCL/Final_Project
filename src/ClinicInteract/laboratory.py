import openai
# 实验室是个函数，接受医生的测试请求，返回测试结果。如果没有相关的测试则返回正常。
from utils import asking_question


class Laboratory:
    """
    Return measurement results based on the doctor's request.
    """
    system_prompt = "You will act as a medical laboratory responsible for responding to doctors' test requests."

    def __init__(self, data_distributor, model="gpt-4o-mini"):
        self.Test_Results = data_distributor.test_result
        self.model = model

    def get_result(self, doctor_request):
        """
        Return the result of the test.
        """
        user_prompt = (f"The doctor requested a test: {doctor_request}\n"
                       f"Here is the list of test results: {self.Test_Results}\nPlease strictly respond with the corresponding test result from the list. "
                       f"Please remember that sometimes the tests requested by the doctor include multiple test results, and you need to output all the included results. For example, "
                       f"for 'Complete Blood Count (CBC)', return items such as 'Hemoglobin (Hb)', 'RBC Count', 'Hematocrit (HCT)', Mean Corpuscular Volume (MCV)', 'Mean Corpuscular Hemoglobin (MCH)', and 'Mean Corpuscular Hemoglobin Concentration (MCHC)'."
                       f"If one of the test is not found in the results, reply with 'Result is normal.' for that test. Do not add any additional information."
                       )

        response = asking_question(self.model, user_prompt, Laboratory.system_prompt)
        print("Lab response:", response)
        return response

if __name__ == "__main__":
    from utils import DataDistributor
    # test
    data_distributor = DataDistributor({"Physical Examination Findings": "On physical examination, her blood pressure is 105/67 mm Hg, the heart rate is 96/min and regular, breathing rate is 23/min, and the pulse oximetry is 96%. An S3 heart sound and rales in the lower right and left lung lobes are heard.",
                                        "Laboratory Results": "A 12-lead ECG shows no significant findings. Echocardiography shows an enlarged left ventricle and left atrium.", "Main Information": "A 43-year-old woman is brought to the emergency department by her brother for severe chest pain. The patient recently lost her husband in a car accident and is still extremely shocked by the event. The patient is stabilized and informed about the diagnosis and possible treatment options.", "answer": "Takotsubo cardiomyopathy"})
    lab = Laboratory(data_distributor)
    result = lab.get_result("Echocardiography")
    print(result)
    result2=lab.get_result("Complete Blood Count (CBC)")
    print(result2)