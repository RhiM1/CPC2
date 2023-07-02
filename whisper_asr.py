from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor, \
    WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer, \
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments
import pandas as pd
import argparse
import torch
import numpy as np
# import torchaudio
# import torchaudio.functional as F
# import torchaudio.transforms as T
from torch.utils.data import Dataset
import tqdm
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, DatasetDict, Dataset, Audio
# from sklearn.model_selection import train_test_split
from data_handling import get_disjoint_val_set
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from process_cpc2_data import get_cpc2_dataset
from constants import DATAROOT

# DATAROOT = "/store/store1/data/clarity_CPC2_data/clarity_data/"
# DATAROOT = "/home/acp20rm/exp/data/clarity_CPC2_data/clarity_data/"
# DATAROOT = "~/exp/data/clarity_CPC2_data/clarity_data/"
# DATAROOT = "~/exp/data/clarity_CPC2_data/clarity_data/"

def get_training_args(args):
    training_args = Seq2SeqTrainingArguments(
        output_dir=f"whisper/{args.run_name}",  # change to a repo name of your choice
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_acc,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.lr,
        warmup_steps=500,
        max_steps=6000,
        # max_steps=2,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        # save_steps=1,
        save_steps=1000,
        # eval_steps=1,
        eval_steps=1000,
        logging_steps=25,
        report_to=["wandb"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        run_name = args.run_name
    )
    return training_args


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"], "attention_mask": feature["attention_mask"]} for feature in features]
        # attention_mask = [{"attention_mask": feature["attention_mask"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
    
def compute_metric_with_args(tokenizer, metric):
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    return compute_metrics


def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    processor = WhisperProcessor.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    data = get_cpc2_dataset(args, processor = processor)

    # data_dict = data.filter(lambda row: row['validation'] == 0).train_test_split(test_size = 0.1)
    # data_dict['dis_val'] = data.filter(lambda row: row['validation'] > 0)
    # data_dict['dis_lis_val'] = Dataset.from_pandas(data[data.validation.isin([1, 3, 5, 7])])
    # data_dict['dis_sys_val'] = Dataset.from_pandas(data[data.validation.isin([2, 3, 6, 7])])
    # data_dict['dis_scene_val'] = Dataset.from_pandas(data[data.validation.isin([4, 5, 6, 7])])

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")


    training_args = get_training_args(args)

    compute_metric_fn = compute_metric_with_args(processor.tokenizer, metric)

    trainer = Seq2SeqTrainer(
        args = training_args,
        model = model,
        train_dataset = data["train"],
        eval_dataset = data["test"],
        data_collator = data_collator,
        compute_metrics = compute_metric_fn,
        tokenizer = processor.feature_extractor
    )

    trainer.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--processed_data_file", help="location of pre-processed datafile" , default="huggingface_data"
    )
    parser.add_argument(
        "--resample_rate", help="wav resample rate" , default=16000
    )
    parser.add_argument(
        "--retain_wav", help="retain the wav signal when extracting features" , default=False, type = bool
    )
    parser.add_argument(
        "--disk_data", help="load extracted data from file" , default=True, type = bool
    )
    parser.add_argument(
        "--data_set_loc", help="location of presaved dataset" , default="feats_data", type = str
    )


    # Whisper 
    parser.add_argument(
        "--whisper_model", help="location of configuation json file", default = "openai/whisper-small"
    )
    parser.add_argument(
        "--whisper_language", help="location of configuation json file", default = "English"
    )
    parser.add_argument(
        "--whisper_task", help="location of configuation json file", default = "transcribe"
    )

    ## Training
    parser.add_argument(
        "--in_json_file", help="JSON file containing the CPC2 test metadata", default = None
    )
    parser.add_argument(
        "--do_train", help="do training", default=False, type=bool
    )
    parser.add_argument(
        "--lr", help="learning rate", default=1e-5, type=float
    )
    parser.add_argument(
        "--gradient_acc", help="learning rate", default=2, type=float
    )
    parser.add_argument(
        "--batch_size", help="batch size" , default=8, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=0, type=int
    )


    # Test data
    parser.add_argument(
        "--test_json_file", help="JSON file containing the CPC2 test metadata", 
    )
    parser.add_argument(
        "--N", help="train split" , default = 1
    )

    # General
    parser.add_argument(
        "--exp_id", help="id for individual experiment", default = "000"
    )
    parser.add_argument(
        "--seed", help="torch seed" , default=1234,
    )
    parser.add_argument(
        "--skip_wandb", help="skip logging via WandB", default=False, action='store_true'
    )

    parser.add_argument(
        "--summ_file", help="train and evaluate on CPC1 data" , default=None
    )

    args = parser.parse_args()

    # Train data
    args.in_json_file = f"{DATAROOT}/metadata/CEC1.train.{args.N}.json"
    args.processed_data_file = f"{DATAROOT}{args.processed_data_file}_N{args.N}"

    args.run_name = f"{args.exp_id}_{args.N}_lr{args.lr}_bs{args.batch_size}_ga{args.gradient_acc}"
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)




