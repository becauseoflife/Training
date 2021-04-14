import json


class LoadData(object):
    def __init__(self, file_name):
        self.data_file_path = "../identification/data/"
        self.data_file_name = file_name + '.json'
        self.data_file = self.data_file_path + self.data_file_name

    def save_data(self, json_data):
        with open(self.data_file, 'w') as file_obj:
            json.dump(json_data, file_obj)

    def load_data(self):
        with open(self.data_file) as file_obj:
            return json.load(file_obj)