import torch
import argparse
import pandas as pd
from models.ni_predictor_models import ffnn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import random
import csv
from scipy.stats import spearmanr

def std_err(x, y):
    return np.std(x - y) / np.sqrt(len(x))


class cal_ffnn(torch.nn.Module):

    def __init__(
            self,
            input_dim = 1, 
            embed_dim = 8,
            output_dim = 1,
            dropout = 0.0,
            activation = torch.nn.ReLU()
            ):
        super().__init__()

        self.ffnn = ffnn(
            input_dim = input_dim,
            embed_dim = embed_dim,
            output_dim = output_dim,
            dropout = dropout,
            activation = activation
        )
        self.sigmoid = torch.nn.Sigmoid()
        

    def forward(self, x):

        x = self.ffnn(x)
        x = self.sigmoid(x)
        
        return x * 100

class pdDataset(Dataset):
 
  def __init__(self, df):
 
    self.predicted = torch.tensor(df.predicted, dtype = torch.float)
    if "correctness" in df:
       self.correctness = torch.tensor(df.correctness, dtype = torch.float)
    else:
       self.correctness = None
       print("Proceeding without correctness values...")
    if "listener" in df:
       self.listener = df.listener
    else:
       self.listener = None
    if "system" in df:
       self.system = df.system
    else:
       self.system = None
    if "scene" in df:
       self.scene = df.scene
    else:
       self.scene = None
 
  def __len__(self):

    return len(self.predicted)
   
  def __getitem__(self,idx):
    correct = None if self.correctness is None else self.correctness[idx].unsqueeze(-1)
    listen = None if self.listener is None else self.listener[idx]
    system = None if self.system is None else self.system[idx]
    scene = None if self.scene is None else self.scene[idx]

    return self.predicted[idx].unsqueeze(-1), correct, listen, system, scene
  
def get_stats(predictions, correctness):
    p_corr = np.corrcoef(predictions,correctness)[0][1]
    s_corr = spearmanr(predictions,correctness)[0]
    std = std_err(predictions, correctness)
    stats_dict = {
        "p_corr": p_corr,
        "s_corr": s_corr,
        "std": std
    }
    return stats_dict


def train_calibration(data, cal_model, optimizer, lossFn, args):

    running_loss = 0
    dataloader = DataLoader(dataset = data, batch_size = args.batch_size, shuffle = True)
    len_dat = len(dataloader)

    for batchID, batch in enumerate(dataloader):

        predicted, correctness, _, _, _ = batch

        optimizer.zero_grad()
        cal_preds = cal_model(predicted)
        loss = lossFn(cal_preds[:, 0], correctness[:, 0])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return cal_model, optimizer, running_loss / len_dat


def apply_calibration(data, cal_model, lossFn, args):

    running_loss = 0
    dataloader = DataLoader(dataset = data, batch_size = args.batch_size, shuffle = False)
    len_dat = len(dataloader)
    all_cal_preds = torch.zeros(len(data), dtype = torch.float)

    for batchID, batch in enumerate(dataloader):
       
        predicted, correctness, lisener, system, scene = batch

        cal_preds = cal_model(predicted)

        all_cal_preds[batchID * args.batch_size:(batchID + 1) * args.batch_size] = cal_preds[:, 0]

        predicted = predicted[:, 0]

        # for pred in cal_preds:
        #    cal_preds_list.append(pred[0].cpu().detach().numpy())
        # print(cal_preds_list)
        # quit()

        if correctness is not None:
            loss = lossFn(cal_preds, correctness)
            running_loss += loss.item()


    return all_cal_preds, running_loss / len_dat


def main(args):

    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    val_file = f"{args.results_file}_val_preds.csv"
    all_dis_file = f"{args.results_file}_all_dis_val_preds.csv"
    dis_file = f"{args.results_file}_dis_val_preds.csv"
    lis_file = f"{args.results_file}_dis_lis_val_preds.csv"
    sys_file = f"{args.results_file}_dis_sys_val_preds.csv"
    scene_file = f"{args.results_file}_dis_scene_val_preds.csv"

    val_df = pd.read_csv(val_file)
    all_dis_df = pd.read_csv(all_dis_file)
    dis_df = pd.read_csv(dis_file)
    lis_df = pd.read_csv(lis_file)
    sys_df = pd.read_csv(sys_file)
    scene_df = pd.read_csv(scene_file)

    val_data = pdDataset(val_df)
    all_dis_data = pdDataset(all_dis_df)
    dis_data = pdDataset(dis_df)
    lis_data = pdDataset(lis_df)
    sys_data = pdDataset(sys_df)
    scene_data = pdDataset(scene_df)

    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    cal_model = cal_ffnn(
        input_dim = 1,
        embed_dim = args.hidden_dim,
        output_dim = 1,
        dropout = args.dropout
    )

    if args.pretrained is not None:
        cal_model.load_state_dict(torch.load(args.pretrained))

    lossFn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        cal_model.parameters(), 
        lr = args.lr, 
        weight_decay = args.weight_decay
    )
    
    _, best_all_dis_loss = apply_calibration(
        data = all_dis_data,
        cal_model = cal_model,
        lossFn = lossFn,
        args = args
    )
    best_epoch = 0

    torch.save(cal_model.state_dict(), f"{args.model_name}.mod")
    torch.save(optimizer.state_dict(), f"{args.model_name}.opt")

    print("\nTraining ----------------------------------")
    if args.do_train:
        

        for epoch in range(args.n_epochs):

            cal_model, optimizer, val_loss = train_calibration(
                data = val_data, 
                cal_model = cal_model,
                optimizer = optimizer,
                lossFn = lossFn,
                args = args
            )

            _, all_dis_loss = apply_calibration(
               data = all_dis_data,
               cal_model = cal_model,
               lossFn = lossFn,
               args = args
            )

            if all_dis_loss < best_all_dis_loss:
                torch.save(cal_model.state_dict(), f"{args.model_name}.mod")
                torch.save(optimizer.state_dict(), f"{args.model_name}.opt")
                best_epoch = epoch + 1

            print(f"Epoch {epoch + 1}: \t {val_loss**0.5:>0.3f} \t {all_dis_loss**0.5:>0.3f}")


    print("\nPredicting --------------------------------")
    print(f"Best epoch: {best_epoch}")

    cal_model.load_state_dict(torch.load(f"{args.model_name}.mod"))

    val_preds, val_loss = apply_calibration(
            data = val_data,
            cal_model = cal_model,
            lossFn = lossFn,
            args = args
        )
    all_dis_preds, all_dis_loss = apply_calibration(
            data = all_dis_data,
            cal_model = cal_model,
            lossFn = lossFn,
            args = args
        )
    dis_preds, dis_loss = apply_calibration(
            data = dis_data,
            cal_model = cal_model,
            lossFn = lossFn,
            args = args
        )
    lis_preds, lis_loss = apply_calibration(
            data = lis_data,
            cal_model = cal_model,
            lossFn = lossFn,
            args = args
        )
    sys_preds, sys_loss = apply_calibration(
            data = sys_data,
            cal_model = cal_model,
            lossFn = lossFn,
            args = args
        )
    scene_preds, scene_loss = apply_calibration(
            data = scene_data,
            cal_model = cal_model,
            lossFn = lossFn,
            args = args
        )
    
    if val_data.correctness is not None:
        base_val_loss = lossFn(val_data.predicted, val_data.correctness)
    else:
        base_val_loss = None

    if all_dis_data.correctness is not None:
        base_all_dis_loss = lossFn(all_dis_data.predicted, all_dis_data.correctness)
    else:
        base_all_dis_loss = None

    if dis_data.correctness is not None:
        base_dis_loss = lossFn(dis_data.predicted, dis_data.correctness)
    else:
        base_dis_loss = None

    if lis_data.correctness is not None:
        base_lis_loss = lossFn(lis_data.predicted, lis_data.correctness)
    else:
        base_lis_loss = None

    if lis_data.correctness is not None:
        base_sys_loss = lossFn(sys_data.predicted, sys_data.correctness)
    else:
        base_sys_loss = None

    if lis_data.correctness is not None:
        base_scene_loss = lossFn(scene_data.predicted, scene_data.correctness)
    else:
        base_scene_loss = None
    
    print()
    print(f"val_loss: \t {base_val_loss**0.5:>0.3f} \t {val_loss**0.5:>0.3f}")
    print(f"all_dis_loss: \t {base_all_dis_loss**0.5:>0.3f} \t {all_dis_loss**0.5:>0.3f}")
    print(f"dis_loss: \t {base_dis_loss**0.5:>0.3f} \t {dis_loss**0.5:>0.3f}")
    print(f"lis_loss: \t {base_lis_loss**0.5:>0.3f} \t {lis_loss**0.5:>0.3f}")
    print(f"sys_loss: \t {base_sys_loss**0.5:>0.3f} \t {sys_loss**0.5:>0.3f}")
    print(f"scene_loss: \t {base_scene_loss**0.5:>0.3f} \t {scene_loss**0.5:>0.3f}")
    print()
    
    val_df["cal_predicted"] = val_preds.cpu().detach().numpy()
    all_dis_df["cal_predicted"] = all_dis_preds.cpu().detach().numpy()
    dis_df["cal_predicted"] = dis_preds.cpu().detach().numpy()
    lis_df["cal_predicted"] = lis_preds.cpu().detach().numpy()
    sys_df["cal_predicted"] = sys_preds.cpu().detach().numpy()
    scene_df["cal_predicted"] = scene_preds.cpu().detach().numpy()

    val_df.to_csv(f"{args.output_dir}/val.csv")
    all_dis_df.to_csv(f"{args.output_dir}/all_dis.csv")
    dis_df.to_csv(f"{args.output_dir}/dis.csv")
    lis_df.to_csv(f"{args.output_dir}/lis.csv")
    sys_df.to_csv(f"{args.output_dir}/sys.csv")
    scene_df.to_csv(f"{args.output_dir}/scene.csv")
    print("Done! -------------------------------------\n")


    with open (args.summ_file, "a") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([args.model_name.split("/")[-1], val_loss**0.5, all_dis_loss**0.5, dis_loss**0.5, lis_loss**0.5, sys_loss**0.5, scene_loss**0.5])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_file", help="location of results files", default = "save/017_N1_WhisperFull_ExLSTM_layers" 
    )    
    parser.add_argument(
        "--output_dir", help="location of output directory", default = None
    )
    parser.add_argument(
        "--summ_file", help="path to write summary results to" , default="cals/CPC2_metrics_cal.csv"
    )
    parser.add_argument(
        "--seed", help="random seed", default=1234, type=int
    )

    # Model
    parser.add_argument(
        "--pretrained", help="pretrained calibration model file", default=None
    )
    parser.add_argument(
        "--hidden_dim", help="calibration model hidden layer dimension", default=8, type=int
    )

    # Training and hyperparameters
    parser.add_argument(
        "--do_train", help="do training", default=True, action='store_false'
    )
    parser.add_argument(
        "--batch_size", help="batch size" , default=1, type=int
    )
    parser.add_argument(
        "--n_epochs", help="number of epochs", default=10, type=int
    )
    parser.add_argument(
        "--lr", help="learning rate", default=0.001, type=float
    )
    parser.add_argument(
        "--weight_decay", help="weight decay", default=0, type=float
    )
    parser.add_argument(
        "--dropout", help="weight decay", default=0, type=float
    )
    parser.add_argument(
        "--grad_clipping", help="gradient clipping", default=1, type=float
    )

    args = parser.parse_args()

    if args.output_dir is None:
       args.output_dir = f"cals/{args.results_file.split('/')[-1]}"

    args.model_name = f"{args.output_dir}/cal_hd{args.hidden_dim}_bs{args.batch_size}_lr{args.lr}_wd{args.weight_decay}_do{args.dropout}"

    main(args)

    