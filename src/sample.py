class Sample:
    sample_id: int
    data: dict
    label: str

    def __init__(self, sample_id, data, label):
        self.sample_id = sample_id
        self.data = data
        self.label = label
