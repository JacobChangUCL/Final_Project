import openai
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
