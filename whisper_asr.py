from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor, \
    WhisperTokenizer, WhisperProcessor, Seq2SeqTrainer, \
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments
import pandas as pd
import argparse
import torch
import numpy as np
from disjoint_val import get_disjoint_val_set
# import torchaudio
# import torchaudio.functional as F
# import torchaudio.transforms as T
from torch.utils.data import Dataset
import tqdm
from torch.utils.data import DataLoader
import datasets
from datasets import load_dataset, DatasetDict, Dataset, Audio
# from sklearn.model_selection import train_test_split
from disjoint_val import get_disjoint_val_set
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import os
import evaluate

DATAROOT = "/store/store1/data/clarity_CPC2_data/clarity_data/"
# DATAROOT = "/home/acp20rm/exp/data/clarity_CPC2_data/clarity_data/"
# DATAROOT = "~/exp/data/clarity_CPC2_data/clarity_data/"
# DATAROOT = "~/exp/data/clarity_CPC2_data/clarity_data/"


def get_training_args():
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-small-hi",  # change to a repo name of your choice
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=6000,
        # max_steps=100,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        # save_steps=50,
        save_steps=1000,
        # eval_steps=50,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )
    return training_args


def prepare_dataset(batch, feature_extractor, tokenizer):
    
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["prompt"]).input_ids

    return batch

def get_cpc2_dataset(args, feature_extractor, tokenizer):

    # if args.disk_data and os.path.exists(DATAROOT + args.data_set_loc):
    #     data = Dataset.load_from_disk(DATAROOT + args.data_set_loc)
    # else:
    data = pd.read_json(args.train_json_file)
    data["subset"] = "CEC1"
    data2 = pd.read_json(args.train_json_file.replace("CEC1","CEC2"))
    data2["subset"] = "CEC2"
    data = pd.concat([data, data2], ignore_index = True)
    data2 = None
    data = data.drop_duplicates(subset = ['signal'], keep = 'last')
    data["predicted"] = np.nan  # Add column to store intel predictions

    wav_loc_list =[]

    for index, row in data.iterrows():
        wav_loc_list.append(f"{DATAROOT}HA_outputs/train.{args.N}/{row.subset}/{row.signal}.wav")

    data['audio'] = wav_loc_list
    data = get_disjoint_val_set(args, data)

    # train_data, val_data = train_test_split(data[data.validation == 0],test_size=0.1)

    data_dict = Dataset.from_pandas(data[data.validation == 0])
    data_dict = data_dict.train_test_split(test_size = 0.1)
    data_dict['dis_val'] = Dataset.from_pandas(data[data.validation == 7])
    data_dict['dis_lis_val'] = Dataset.from_pandas(data[data.validation.isin([1, 3, 5, 7])])
    data_dict['dis_sys_val'] = Dataset.from_pandas(data[data.validation.isin([2, 3, 6, 7])])
    data_dict['dis_scene_val'] = Dataset.from_pandas(data[data.validation.isin([4, 5, 6, 7])])

    data_dict = data_dict.cast_column("audio", Audio(sampling_rate = 16000, mono = True))

    fnKwargs = {
        "feature_extractor": feature_extractor,
        "tokenizer": tokenizer,
    }
    data_dict = data_dict.map(prepare_dataset, num_proc = 4, fn_kwargs = fnKwargs)

        # if args.disk_data:
        #     data.save_to_disk(DATAROOT + args.data_set_loc)

    return data_dict


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
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

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.whisper_model
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )

    data = get_cpc2_dataset(args, feature_extractor = feature_extractor, tokenizer = tokenizer)

    processor = WhisperProcessor.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


    metric = evaluate.load("wer")

    model = WhisperForConditionalGeneration.from_pretrained(args.whisper_model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print(model)

    training_args = get_training_args()

    compute_metric_fn = compute_metric_with_args(tokenizer, metric)

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

    # trainer.save_metrics()

    # eval_metrics_test = trainer.evaluate(data["test"])
    # print(f"Validation metrics:\n{eval_metrics_test}")
    # eval_metrics_dis_val= trainer.evaluate(data["dis_val"])
    # print(f"Disjoint validation metrics:\n{eval_metrics_dis_val}")
    # eval_metrics_dis_lis_val= trainer.evaluate(data["dis_lis_val"])
    # print(f"Disjoint listener validation metrics:\n{eval_metrics_dis_lis_val}")
    # eval_metrics_dis_sys_val= trainer.evaluate(data["dis_sys_val"])
    # print(f"Disjoint system validation metrics:\n{eval_metrics_dis_sys_val}")
    # eval_metrics_dis_scene_val= trainer.evaluate(data["dis_scene_val"])
    # print(f"Disjoint scene validation metrics:\n{eval_metrics_dis_scene_val}")
    # eval_metrics_train = trainer.evaluate(data["train"])
    # print(f"Training data metrics:\n{eval_metrics_train}")
    # eval_metrics_all = trainer.evaluate(data)
    # print(f"Training data metrics:\n{eval_metrics_all}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
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
        "--train_json_file", help="JSON file containing the CPC2 test metadata", default = None
    )
    parser.add_argument(
        "--do_train", help="do training", default=False, type=bool
    )
    parser.add_argument(
        "--lr", help="learning rate", default=999, type=float
    )
    parser.add_argument(
        "--batch_size", help="batch size" , default=999, type=int
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
        "--exp_id", help="id for individual experiment"
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
    args.train_json_file = f"{DATAROOT}/metadata/CEC1.train.{args.N}.json"


    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)




