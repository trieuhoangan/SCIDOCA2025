import os
import json
import argparse
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
from rank_bm25 import BM25Okapi
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

def load_data(data_path):
    data_files = os.listdir(data_path)
    filenames = []
    
    for file in data_files:
        file_id = file.split(".")[0]
        if file_id not in filenames:
            filenames.append(file_id)
    full_data = merge_input_label_to_list(data_path, filenames)
    return full_data

def find_correct_citation(label_data):
    sent_corrects = []
    # print(label_data["correct_citation"])
    for sent in label_data["correct_citation"]:
        sent_corrects.extend(label_data["correct_citation"])
    return list(set(sent_corrects))

def merge_input_label_to_list(data_path, filenames):
    data = []
    for index, filename in enumerate( filenames):
        if index > 1000:
            break
        input_full_path = os.path.join(data_path, f"{filename}.in")
        label_full_path = os.path.join(data_path, f"{filename}.label")
        with open(input_full_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(label_full_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        print(label_data["correct_citation"])
        data.append({
            "text": input_data["text"], 
            "citation_candidates": input_data["citation_candidates"],
            "candidate_bib_entries": input_data["bib_entries"],
            "correct_citation": find_correct_citation(label_data)
        })
    return data


def merge_input_label(data_path, filenames):
    data = {}
    for filename in filenames:
        input_full_path = os.path.join(data_path, f"{filename}.in")
        label_full_path = os.path.join(data_path, f"{filename}.label")
        with open(input_full_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(label_full_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        data[filename] = {
            "text": input_data["text"], 
            "citation_candidates": input_data["citation_candidates"],
            "candidate_bib_entries": input_data["bib_entries"],
            "correct_citation": label_data["correct_citation"]
        }
    return data

def split_data(dataset):
    dataset = dataset.train_test_split(test_size = 0.2, shuffle = False)
    trainset = dataset["train"]
    test_dataset = dataset["test"]
    trainset = trainset.train_test_split(test_size = 0.2, shuffle= False)
    train_dataset = trainset["train"]
    valid_dataset = trainset["test"]

    return train_dataset, valid_dataset, test_dataset

def build_corpus_from_candidates_abstract(candidates):
    id_list = list(candidates.keys())
    corpus = []
    for key in candidates:
        corpus.append("{}. {}".format(candidates[key]['title'],candidates[key]['abstract']))
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    # print(len(tokenized_corpus))
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, id_list

def do_retrieve(sentences, candidates):
    complete_sent = " ".join(sentences)
    tokenized_query = complete_sent.split(" ")
    bm25, id_list = build_corpus_from_candidates_abstract(candidates)
    doc_scores = bm25.get_scores(tokenized_query)
    print("retrieve scores ", doc_scores)
    # correct_index = doc_scores.index(max(doc_scores))
    correct_index = np.argmax(doc_scores)
    print("id list", id_list)
    predict_citation = [id_list[correct_index]]
    return predict_citation

def main(args):
    # Load data
    print("loading data")
    full_data = load_data(args.data_dir)
    print("finish loading data")
    print(len(full_data))
    # dataset = Dataset.from_list(full_data)
    # print(dataset)
    # train_dataset, valid_dataset, test_dataset = split_data(dataset)
    # print(dataset)
    count=0
    correct = 0
    print("start retrieving")
    for row in full_data:
        count+=1
        
        prediction = do_retrieve(row["text"],row["candidate_bib_entries"])
        print("row correct_citation ",row["correct_citation"])
        print("row citation_candidates ",row["citation_candidates"])
        # for co
        print("=="*100)
        if prediction == row["correct_citation"]:
            correct+=1

    print("accuracy: ", correct/count)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/", help="Path containing data")
    parser.add_argument('--ft_model_id', type=str, default=None, help='fintuned model id for saving after train it')
    parser.add_argument("--model_name_or_path", type=str, default="google/mt5-xl", help="mT5-XL model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./mt5_xl_translation", help="Where to save model")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--do_train", action="store_true", help="Whether to train")
    parser.add_argument("--do_eval", action="store_true", help="Whether to evaluate on valid")
    parser.add_argument("--do_predict", action="store_true", help="Whether to predict on test")
    args = parser.parse_args()
    main(args)