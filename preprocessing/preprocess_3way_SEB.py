import xml.etree.ElementTree as et
import os
import nltk

train_root = '../../datasets/training/sciEntsBank'
test_root = '../../datasets/testing/sciEntsBank'


def load_data(path):
    data = []
    files = os.listdir(path)
    for file in files:
        root = et.parse(path + '/' + file).getroot()
        data.append(
            {
                "id": root.get('id'),
                "question": root[0].text,
                "ref_ans": [x.text for x in root[1]],
                "stud_ans": [[x.text, x.get('accuracy')] for x in root[2]]
            }
        )
    return data


train_data = load_data(train_root)
test_ua, test_ud, test_uq = [load_data(test_root + '/' + x) for x in os.listdir(test_root)]
print(train_data)

