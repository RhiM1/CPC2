from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
import pandas as pd
import argparse
import torch
import numpy as np
from disjoint_val import get_disjoint_val_set
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

# DATAROOT = "/store/store1/data/clarity_CPC2_data/clarity_data/"
DATAROOT = "~/data/clarity_CPC2_data/clarity_data/"

def audio_pipeline(path,fs=32000):
    resampler = T.Resample(32000, 16000).to(args.device)
    # wavs = sb.dataio.dataio.read_audio_multichannel(path)   
    print(path) 
    wavs, sample_rate = torchaudio.load(path)
    print(f"sample rate: {sample_rate}")
    wavs_l = wavs[:,:,0]
    wavs_r = wavs[:,:,1]
    wavs_l = resampler(wavs_l)
    wavs_r = resampler(wavs_r)
    return wavs

def get_cpc2_dataset(args):
    
    data = pd.read_json(args.train_json_file)
    data["subset"] = "CEC1"
    data2 = pd.read_json(args.train_json_file.replace("CEC1","CEC2"))
    data2["subset"] = "CEC2"
    data = pd.concat([data, data2])

    wav_files = []
    for index, row in data.iterrows():
        # path = f"{DATAROOT}HA_outputs/train.{args.N}/{row.subset}/{row.scene}_{row.listener}_{row.system}.wav"
        # path = f"{DATAROOT}HA_outputs/signals/{row.subset}/{row.signal}.wav"
        # path = f"~/data/clarity_CPC2_data/clarity_data/HA_outputs/train.1/CEC1/S08547_L0239_E001.wav"
        path = f"~/data/clarity_CPC2_data/clarity_data/HA_outputs/signals/CEC1/S08547_L0239_E001.wav"
        path = args.train_json_file
        # path = f"~/data/clarity_CPC2_data/clarity_data/HA_outputs/signals/train.{args.N}/CEC1/S08547_L0239_E001.wav"
        # print(path)
        wav = audio_pipeline(path)
        print(wav)
        quit()

    data['wav_files'] = wav_files
    print(data.wav_files)
    


    # {"func": lambda l: audio_pipeline("%s/clarity_data/HA_outputs/%s/%s.wav"%(args.dataroot,theset,l),32000),
    #         "takes": ["signal"],
    #         "provides": "wav"}

    data["predicted"] = np.nan  # Add column to store intel predictions
    data = get_disjoint_val_set(args, data)

    # print(data)

    return data



def main(args):

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_model)
    tokenizer = WhisperTokenizer.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )
    processor = WhisperProcessor.from_pretrained(
        args.whisper_model, 
        language = args.whisper_language, 
        task = args.whisper_task
    )

    data = get_cpc2_dataset(args)
    quit()
    print(data)

    input_str = data.iloc[0].prompt
    print(input_str)

    tokens = tokenizer(input_str)
    labels = tokens.input_ids
    decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
    decoded_str = tokenizer.decode(labels, skip_special_tokens=True)
    print(f"Input:                 {input_str}")
    print(f"Decoded w/ special:    {decoded_with_special}")
    print(f"Decoded w/out special: {decoded_str}")
    print(f"Are equal:             {input_str == decoded_str}")

    print(tokens)

    # configuration = WhisperConfig
    # model = WhisperModel(configuration)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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
        "--seed", help="torch seed" , default=999,
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




