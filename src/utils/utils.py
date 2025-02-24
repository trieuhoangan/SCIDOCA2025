import json
import os
from datasets import Dataset
import evaluate
import numpy as np

def load_data(data_path):
    data_files = os.listdir(data_path)
    filenames = []
    for file in data_files:
        if len(filenames) > 100:
            break
        file_id = file.split(".")[0]
        if file_id not in filenames:
            filenames.append(file_id)
    full_data = merge_input_label(data_path, filenames)
    return full_data

def merge_input_label(data_path, filenames):
    data = []
    for filename in filenames:
        input_full_path = os.path.join(data_path, f"{filename}.in")
        label_full_path = os.path.join(data_path, f"{filename}.label")
        with open(input_full_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(label_full_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        data.append({
            "id": filename, 
            "text": input_data["text"], 
            "citation_candidates": input_data["citation_candidates"],
            "candidate_bib_entries": input_data["bib_entries"],
            "correct_citation": label_data["correct_citation"]
        })
    return data

def preprocess_for_classification(data):
    processed_data = []
    sum_label=0
    for data_content in data:
        id = data_content["id"]
        # data_content = data[id]
        # merged_query_text = " ".join(data_content["text"])
        merged_query_text = data_content["text"]
        correct_candidates = data_content["correct_citation"]
        # print("data_content['correct_citation'] ", data_content["correct_citation"])
        # print("data_content['candidate_bib_entries'] ", data_content["candidate_bib_entries"])
        # print("data_content['text'] ", data_content["text"])
        # print("merged_query_text ", merged_query_text)
        # for label in data_content["correct_citation"]:
        #     correct_candidates.extend(label)
        correct_candidates = list(set(correct_candidates))
        for candidate in data_content["candidate_bib_entries"]:
            # print("candidate ", candidate)
            # print("correct_candidates ", correct_candidates)
            merged_input = "classify: {} . {} . {}".format(merged_query_text, data_content["candidate_bib_entries"][candidate]["title"], data_content["candidate_bib_entries"][candidate]["abstract"])
            
            if candidate in correct_candidates:
                current_label = 1
            else:
                current_label = 0
            processed_data.append(
                {
                    "query_id": id,
                    "candidate_id":candidate,
                    "text": merged_input,
                    "label":current_label 
                }
            )

            sum_label+=current_label
    print(sum_label)
    print(len(processed_data))
    return processed_data



def split_list_by_ratio(data, ratio):
    """
    Splits `data` into two sub-lists based on a single ratio value.

    The second sub-list will have `ratio * len(data)` elements (rounded down),
    and the first sub-list will have the remaining elements.

    :param data: The list to be split.
    :param ratio: A float between 0 and 1 indicating the fraction of data
                  that should go into the second sub-list.
    :return: A tuple of two lists, (sub_list_1, sub_list_2).
    """
    if not (0 <= ratio <= 1):
        raise ValueError("ratio must be between 0 and 1 (inclusive).")
    split_index = int(len(data) * (1 - ratio))
    return data[:split_index], data[split_index:]

def split_data(dataset):
    dataset = dataset.train_test_split(test_size = 0.2, shuffle = False)
    trainset = dataset["train"]
    test_dataset = dataset["test"]
    trainset = trainset.train_test_split(test_size = 0.2, shuffle= False)
    train_dataset = trainset["train"]
    valid_dataset = trainset["test"]

    return train_dataset, valid_dataset, test_dataset

def split_list_data(data, test_ratio):
    # test_ratio = 0.2
    trainset, test_dataset = split_list_by_ratio(data, test_ratio)
    train_dataset, valid_dataset = split_list_by_ratio(trainset,test_ratio)

    return train_dataset, valid_dataset, test_dataset

def load_and_split_data_for_training(datapath):
    '''This function is to load data from data dir and split into train, valid and test dataset 
    Data is splitted base in number of file instead of number of (query + abstract)
    each file have 1 query and n candidates -> n (query + abstract)
    For example: data dir have 100 files
    -> test set = (100* test_ratio) files
    -> valid set = (100 -  test_set)*(test_ratio) files
    -> train set = (100 - test_set - valid_set) files
    '''
    data_files = os.listdir(datapath)
    filenames = []
    for file in data_files:
        if len(filenames) > 5000:
            break
        file_id = file.split(".")[0]
        if file_id not in filenames:
            filenames.append(file_id)
    test_ratio = 0.2
    train_filenames, valid_filenames, test_filenames = split_list_data(filenames, test_ratio)
    train_data = merge_input_label(datapath, train_filenames)
    valid_data = merge_input_label(datapath, valid_filenames)
    test_data = merge_input_label(datapath, test_filenames)

    train_data_list = preprocess_for_classification(train_data)
    valid_data_list = preprocess_for_classification(valid_data)
    test_data_list  = preprocess_for_classification(test_data)

    train_dataset = Dataset.from_list(train_data_list)
    valid_dataset = Dataset.from_list(valid_data_list)
    test_dataset = Dataset.from_list(test_data_list)
    return train_dataset, valid_dataset, test_dataset

def load_data_for_testing(datapath):
    data_files = os.listdir(datapath)
    filenames = []
    for file in data_files:
        file_id = file.split(".")[0]
        if file_id not in filenames:
            filenames.append(file_id)
    data = merge_input_label(datapath, filenames)
    dataset = preprocess_for_classification(data)
    infer_dataset = Dataset.from_list(dataset)
    return infer_dataset,infer_dataset,infer_dataset

def calculate_f1_score(preds, labels):
    correct = 0
    n_retrived = 0
    n_relevant = 0

    coverages = []

    for pred, label in list(zip(preds, labels)):
        
        # target = row['target']
        # preds = row['candidates']
        coverages.append(len(pred))

        n_retrived += len(pred)
        n_relevant += len(label)
        for prediction in preds:
            if prediction in label:
                correct += 1

    precision = correct / n_retrived
    recall = correct / n_relevant

    print(f"Average # candidates: {np.mean(coverages)}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {2 * precision * recall / (precision + recall)}")
