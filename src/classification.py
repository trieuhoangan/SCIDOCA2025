import os
import json
import argparse
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

def load_data(data_path):
    data_files = os.listdir(data_path)
    filenames = []
    for file in data_files:
        file_id = file.split(".")[0]
        if file_id not in filenames:
            filenames.append(file_id)
    full_data = merge_input_label(data_path, filenames)
    return full_data

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

def preprocess_for_classification(data):
    processed_data = []
    for id in data:
        data_content = data[id]
        merged_query_text = " ".join(data_content["text"])
        correct_candidates = []
        for label in data_content["correct_citation"]:
            correct_candidates.extend(label)
        correct_candidates = list(set(correct_candidates))
        for candidate in data_content["candidate_bib_entries"]:
            print(candidate)
            merged_input = "{} . {} . {}".format(merged_query_text, data_content["candidate_bib_entries"][candidate]["title"], data_content["candidate_bib_entries"][candidate]["abstract"])
            if candidate in correct_candidates:
                current_label = 1
            else:
                current_label = 0
            processed_data.append(
                {
                    "text": merged_input,
                    "label":current_label 
                }
            )
    return processed_data

def split_data(dataset):
    dataset = dataset.train_test_split(test_size = 0.2, shuffle = False)
    trainset = dataset["train"]
    test_dataset = dataset["test"]
    trainset = trainset.train_test_split(test_size = 0.2, shuffle= False)
    train_dataset = trainset["train"]
    valid_dataset = trainset["test"]

    return train_dataset, valid_dataset, test_dataset



def main(args):
    # Load data
    full_data = load_data(args.data_dir)
    data_for_clasification = preprocess_for_classification(full_data)
    dataset = Dataset.from_list(data_for_clasification)
    print(dataset)
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path, 
            num_labels=2,
            ignore_mismatched_sizes=True
    )
    
    # preprocess data
    accuracy = evaluate.load("accuracy")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)
    
    
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    print(tokenized_dataset)
    train_dataset, valid_dataset, test_dataset = split_data(tokenized_dataset)

    # train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
   
    # eval
    if args.do_train:
        trainer.train()
    if args.do_eval:
        if "label" in valid_dataset.features:
            valid_dataset = valid_dataset.remove_columns("label")
        predictions = trainer.predict(valid_dataset, metric_key_prefix="predict").predictions
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")

    if args.do_predict:
        if "label" in test_dataset.features:
            test_dataset = test_dataset.remove_columns("label")
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")
    

    return 1

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