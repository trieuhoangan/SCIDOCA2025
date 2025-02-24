import os
import json
import argparse
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics import f1_score
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5

from transformers import T5ForConditionalGeneration

def load_data(data_path):
    data_files = os.listdir(data_path)
    filenames = []
    
    for file in data_files:
        if len(filename) == 1000:
            continue
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

def get_top_n_indices(values, n):
    """
    Return the indices of the top n highest elements in the list 'values'.
    
    :param values: A list of float numbers.
    :param n: The number of top elements to find.
    :return: A list of indices corresponding to the top n highest values (descending order).
    """
    
    # Get a list of all indices, sorted by their corresponding values in descending order
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    
    # Slice the first 'n' indices
    return sorted_indices[:n]

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

def do_retrieve(sentences, candidates):
    complete_sent = " ".join(sentences)
    tokenized_query = complete_sent.split(" ")
    # bm25, id_list = build_corpus_from_candidates_abstract(candidates)
    # id_list = list(candidates.keys())
    id_list = []
    corpus = []
    for key in candidates:
        id_list.append(key)
        corpus.append("{}. {}".format(candidates[key]['title'],candidates[key]['abstract']))
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    # print(len(tokenized_corpus))
    bm25 = BM25Okapi(tokenized_corpus)
    
    doc_scores = bm25.get_scores(tokenized_query)
    
    preds = []

    print("retrieve scores ", doc_scores)
    top_n = get_top_n_indices(doc_scores, 3)
    preds = [id_list[i] for i in top_n]

    # correct_index = doc_scores.index(max(doc_scores))
    # correct_index = np.argmax(doc_scores)
    
    # text_preds =  bm25.get_top_n(tokenized_query, corpus, n=5)
    # for pred in text_preds:
    #     pred_index = corpus.index(pred)
    #     preds.append(id_list[pred_index])
    print("id list", id_list)
    # predict_citation = [id_list[correct_index]]
    return preds

def do_retrieve_with_monot5(reranker, query , candidate_docs):

    ranking_scores = reranker.rescore(Query(query), [Text("{} {}".format(candidate_docs[p]['title'], candidate_docs[p]['abstract']), {'docid': p}, 0) for p in candidate_docs])
    ranking_scores_ls = [r.score for r in ranking_scores]
    print(ranking_scores_ls)


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

    #### Load model
    model_name = "./monoT5_model/monot5-large-msmarco-10k"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    reranker = MonoT5(model=model) 

    preds = []
    labels = []
    print("start retrieving")
    for row in full_data:
        # count+=1
        
        # predictions = do_retrieve(row["text"],row["candidate_bib_entries"])
        # preds.append(predictions)
        labels.append(row["correct_citation"])
        print("row citation_candidates ",row["citation_candidates"])
        print("row correct_citation ",row["correct_citation"])
        print("prediction ", predictions)
        # for co
        print("=="*100)
        do_retrieve_with_monot5(reranker, row["text"] , row["candidate_bib_entries"])
        break
        # if prediction == row["correct_citation"]:
        #     correct+=1
    # calculate_f1_score(preds, labels)
    # print("accuracy: ", correct/count)

    

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