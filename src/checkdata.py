import os
import json
data_path="/home/s2320037/SCIDOCA/data/data_train/task3"

data_files = os.listdir(data_path)

samples = data_files[:30]
print(samples)
full_path=[os.path.join(data_path,filename) for filename in samples]
# for path in full_path:
path = full_path[1]
print(path)
with open(path,'r',encoding='utf-8') as f:
    # lines = f.readlines()
    # print("\n".join(lines))
    # print(path)
    data = json.load(f)

print(data.keys())
# print(data['text'])
# print(len(data['text']))
# print(data['citation_candidates'])
print(data['correct_citation'])
print(data["bib_entries"])

