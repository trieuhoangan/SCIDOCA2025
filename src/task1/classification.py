import os
import sys
sys.path.append("/home/s2320037/SCIDOCA/SCIDOCA2025/src/utils")
# sys.path.append("/home/sam/pythonModules/module2")

from utils import load_and_split_data_for_training, load_data_for_testing
import json
import argparse
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import evaluate
import numpy as np
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# import sys
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.dirname(SCRIPT_DIR))
def main(args):

    
    if args.do_inference:
        train_dataset, valid_dataset, test_dataset = load_data_for_testing(args.data_dir)
    else:
        train_dataset, valid_dataset, test_dataset = load_and_split_data_for_training(args.data_dir)
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
    
    
    # tokenized_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset =  train_dataset.map(preprocess_function, batched=True)
    valid_dataset =  valid_dataset.map(preprocess_function, batched=True)
    test_dataset  =  test_dataset.map(preprocess_function, batched=True)
    # train_dataset, valid_dataset, test_dataset = split_data(tokenized_dataset)
    
    print(test_dataset)
    # train
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=0.01,
        # eval_strategy="epoch",
        save_total_limit=1,
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

        qids = test_dataset["query_id"]         # keep query_ids in a list
        cids = test_dataset["candidate_id"]     # keep candidate_ids in a list
        labels = None

        if "label" in test_dataset.features:
            labels = test_dataset["label"]
            test_dataset = test_dataset.remove_columns("label")


        predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
    
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tquery_id\tcandidate_id\tcorrect_label\tprediction\n")
                for index, pred_label in enumerate(predictions):
                # If labels were present in the original dataset, use them;
                # otherwise, set correct_label to 'N/A'
                    correct_label = labels[index] if labels is not None else "N/A"
                    writer.write(
                        f"{index}\t{qids[index]}\t{cids[index]}\t{correct_label}\t{pred_label}\n"
                    )
    

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
    parser.add_argument("--do_inference", action="store_true", default=False,help="Whether to train")
    parser.add_argument("--do_train", action="store_true", help="Whether to train")
    parser.add_argument("--do_eval", action="store_true", help="Whether to evaluate on valid")
    parser.add_argument("--do_predict", action="store_true", help="Whether to predict on test")
    args = parser.parse_args()
    main(args)