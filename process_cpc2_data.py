from transformers import WhisperProcessor
import pandas as pd
import argparse
import torch
import numpy as np
from disjoint_val import get_disjoint_val_set
from torch.utils.data import Dataset
import datasets
from datasets import load_dataset, Dataset, Audio
import os
from torchaudio.transforms import SlidingWindowCmn

from constants import DATAROOT, DATAROOT_CPC1


def prepare_dataset(batch, processor, cmvn = None):
    
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    # print(f"audio shape: {audio['array'].shape}")
    # compute log-Mel input features from input audio array 
    processed_audio = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"], return_attention_mask = True)
    # print(processed_audio)
    # print(f"attention_mask shape: {processed_audio.attention_mask.shape}")
    batch["input_features"] = processed_audio.input_features[0]
    batch["attention_mask"] = processed_audio.attention_mask
    # print(processed_audio.input_features.shape)
    # print(processed_audio.input_features[0])
    # print()
    if cmvn is not None:
        # print(f"new_audio 1 size: {new_audio.size()}")
        batch["cmvn"] = cmvn(torch.tensor(batch["input_features"])).numpy()
        # print(f"batch['cmvn']: {batch['cmvn'].shape}")
        # print(f"input: {len(batch['input_features'])}, {batch['input_features'].shape}, {batch['input_features'].min()}, {batch['input_features'].max()}\n{batch['input_features']}")
        # print(f"cmvn : {len(batch['cmvn'])}, {batch['cmvn'].shape}\n{batch['cmvn']}\n")
    else:
        new_audio = audio["array"]

    # encode target text to label ids 
    batch["labels"] = processor.tokenizer(batch["prompt"]).input_ids

    return batch





def get_cpc2_dataset(args, processor, save_to_disk = False, return_pd = False, debug = False):

    print(f"processed data file location: {args.processed_data_file}")

    if os.path.exists(args.processed_data_file):
        print("Loading pre-processed data from disk...")
        data_dict = datasets.load_from_disk(args.processed_data_file)
    else:
        print("Processing data...")
        data = pd.read_json(args.in_json_file)
        # data = load_dataset('json', data_files = args.in_json_file)
        data["subset"] = "CEC1"
        data2 = pd.read_json(args.in_json_file.replace("CEC1","CEC2"))
        # data2 = load_dataset('json', data_files = args.in_json_file.replace("CEC1","CEC2"))
        data2["subset"] = "CEC2"
        data = pd.concat([data, data2], ignore_index = True)
        data2 = None
        data = data.drop_duplicates(subset = ['signal'], keep = 'last')

        wav_loc_list =[]

        for index, row in data.iterrows():
            wav_loc_list.append(f"{DATAROOT}HA_outputs/train.{args.N}/{row.subset}/{row.signal}.wav")

        data['audio'] = wav_loc_list
        data = get_disjoint_val_set(args, data)

        if return_pd:
            return data

        data_dict = Dataset.from_pandas(data[data['validation'] == 0]).train_test_split(test_size = 0.1)
        data_dict['dis_val'] = Dataset.from_pandas(data[data['validation'] > 0])

        if debug: # use minimal dataset for debugging
            data_dict['test'] = Dataset.from_dict(data_dict['test'][0:10])
            data_dict['train'] = Dataset.from_dict(data_dict['train'][0:10])
            data_dict['dis_val'] = Dataset.from_dict(data_dict['dis_val'][0:10])

        # data = Dataset.from_pandas(data)
        data_dict = data_dict.cast_column("audio", Audio(sampling_rate = 16000, mono = True))



        if args.use_cmvn:
            cmvn = SlidingWindowCmn(cmn_window = 10000, center = True, norm_vars = True)
        else:
            cmvn = None

        fnKwargs = {"processor": processor, "cmvn": cmvn}
        data_dict = data_dict.map(prepare_dataset, num_proc = 1, fn_kwargs = fnKwargs)

        data_dict = data_dict.remove_columns(['audio'])

        if save_to_disk:

            data_dict.save_to_disk(args.processed_data_file)

    return data_dict


def main(args):
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    processor = WhisperProcessor.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )
    processor.feature_extractor.chunk_length = 10
    processor.feature_extractor.n_samples = processor.feature_extractor.chunk_length * processor.feature_extractor.sampling_rate
    processor.feature_extractor.nb_max_frames = processor.feature_extractor.n_samples // processor.feature_extractor.hop_length
    

    data = get_cpc2_dataset(args, processor = processor, save_to_disk = True, return_pd = False, debug = True)

    print(data)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processed_data_file", help="location of pre-processed datafile" , default=f"huggingface_data"
    )
    parser.add_argument(
        "--resample_rate", help="wav resample rate" , default=16000
    )
    # Test data
    parser.add_argument(
        "--test_json_file", help="JSON file containing the CPC2 test metadata", 
    )
    parser.add_argument(
        "--N", help="train split" , default = 1
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
    parser.add_argument(
        "--use_cmvn", help="include cepstral mean and variance normalisation" , default=False, action='store_true'
    )

    args = parser.parse_args()

    # Train data
    args.in_json_file = f"{DATAROOT}/metadata/CEC1.train.{args.N}.json"
    if args.use_cmvn:
        args.processed_data_file = f"{DATAROOT}{args.processed_data_file}_N{args.N}_cmvn{int(args.use_cmvn)}"
    else:
        args.processed_data_file = f"{DATAROOT}{args.processed_data_file}_N{args.N}"

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(args)




