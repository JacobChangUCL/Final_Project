class PhysicalExamination:
    def __init__(self, data_distributor):
        self.exam_result = data_distributor.physical_examination_findings

    def get_result(self):
        return self.exam_result
