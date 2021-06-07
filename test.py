from models.Informer.model import Informer, InformerStack
from models.LeNet5.lenet import LeNet5
from models.LSTM.lstm import LSTM
import dataloader
import utils.utils as util
from utils.metrics import MSE
import time
import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser(description='Fire detection')

parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD), reformer]')

parser.add_argument('--epochs', default=200, type=int, help='epoch (default: 200)')
parser.add_argument('--batch_size', default=2048, type=int, help='batch size (default: 1024)')
# parser.add_argument('--save-root', default='./exp-results-hidden32/', type=str, help='save root')
parser.add_argument('--print_freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--hidden_size', default=32, type=int, help='hidden size (default: 128)')
parser.add_argument('--traindir',default = './data', type=str, help='train data path')
# parser.add_argument('--testdir',default = '/daintlab/data/sigkdd2021/PhaseII/testset', type=str, help='test data path')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu number')

parser.add_argument('--seq_len', type=int, default=100, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=50, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=25, help='prediction sequence length')

parser.add_argument('--enc_in', type=int, default=1, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=64, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true',default = False, help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='file list')
parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')




parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
#             'reformer':ReformerLM,
            'LeNet5' : LeNet5,
            'lstm' : LSTM,
        }

if args.model=='informer' or args.model=='informerstack':
            e_layers = args.e_layers if args.model=='informer' else args.s_layers ################################
            net = model_dict[args.model](
                args.enc_in,
                args.dec_in,
                args.c_out, 
                args.seq_len, 
                args.label_len,
                args.pred_len, 
                args.factor,
                args.d_model, 
                args.n_heads, 
                e_layers, # self.args.e_layers,
                args.d_layers, 
                args.d_ff,
                args.dropout, 
                args.attn,
                args.embed,
                args.activation,
                args.output_attention,
                args.distil,
                args.mix
            ).float().cuda()

elif args.model == 'reformer':
            net = model_dict[args.model](
                    dim = 512,
                    depth = 6,
                    max_seq_len = args.seq_len,
                    #num_tokens = 256,
                    heads = 8,
                    bucket_size = 64,
                    n_hashes = 4,
                    ff_chunks = 10,
                    lsh_dropout = 0.1,
                    weight_tie = True,
                    causal = True,
                    n_local_attn_heads = 4,
                    use_full_attn = False 
                    )
            
elif args.model == 'LeNet5':
            net = model_dict[args.model](
                    seq_len = args.seq_len,
                    label_len = args.label_len,
                    pred_len=args.pred_len
            ).float().cuda()
            
elif args.model == 'lstm':
            net = model_dict[args.model](
                    n_hidden = 512,
                    seq_len = args.seq_len,
                    pred_len=args.pred_len
            ).float().cuda()
            
criterion = nn.MSELoss().cuda()

def get_global_values(df):
    col = df.columns
    min_value = min(df[col[0]])
    max_value = max(df[col[0]])

    return min_value, max_value

def inverse(x, mini, maxi):
    output = mini + x*(maxi - mini)
    return output

def evaluate(loader, model, criterion):
    batch_time = util.AverageMeter()
    losses = util.AverageMeter()
    end = time.time()
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
            for i, (inputs,target) in enumerate(loader):
                
                inputs = inputs.float().cuda()
                target = target.float().cuda()

                if args.padding==0:
                    dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
                elif args.padding==1:
                    dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
                
                dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
                
                if args.output_attention:
                    outputs, attens = net(inputs, dec_inp)
                    target = target[:,-args.pred_len:,0:].cuda()
                else:
                    outputs = net(inputs,args)
                preds = np.append(preds,outputs[:,-1,:].detach().cpu().numpy())
                trues = np.append(trues,target[:,-1,:].detach().cpu().numpy())
            
                batch_time.update(time.time() - end)
                end = time.time()
                if i % args.print_freq == 0:
                    print(': [{0}/{1}]\t'
                                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        i, len(loader), batch_time=batch_time, loss=losses))
            preds = np.array(preds)
            trues = np.array(trues)
            print('test shape:', preds.shape, trues.shape)

    return  preds, trues

file_list = sorted(os.listdir(args.traindir))
data_list = [file for file in file_list if file.endswith(".csv")]

for i in range(len(data_list)):
    if data_list[i] == 'AMD.csv' or data_list[i] == 'NVDA.csv':
            data_type = 'stock'
            type_dict = {
                    'type1': [35, 14, 7],
                    'type2': [100, 50, 1],
                    'type3': [100, 50, 30],
                }
    else :
#             data_type = 'elec'
        type_dict = {                    
            'type1': [168, 72, 24],
            'type2': [300, 100, 1],
            'type3': [300, 100, 50],
                }
    for key in type_dict.keys() : 

        args.seq_len = type_dict[key][0]
        args.label_len = type_dict[key][1]
        args.pred_len = type_dict[key][2]
            
        args.save_root = f'./exp-results/{args.model}-{data_type}-{key}/'

        save_path = os.path.join(args.save_root, data_list[i])

        test_dataset = dataloader.loader(args.traindir,data_list[i],
                                            seq_size = type_dict[key],loader_type = 'test',args = args)

        test_loader = DataLoader(test_dataset,
                                    shuffle=False, 
                                    batch_size=args.batch_size, 
                                    pin_memory=False)


        state_dict = torch.load(f'{save_path}/model_200.pth')
        net.load_state_dict(state_dict)
        pred, trues = evaluate(test_loader, net, criterion)

        in_losses = MSE(pred, trues)

        plt.figure(figsize=(64, 16))
        plt.plot(trues, color = 'blue',alpha = 0.5, label = 'input')
        plt.plot(pred, color = 'red',alpha = 0.5 ,label = 'output')
        plt.legend(['input', 'output'])
        plt.savefig(f'{save_path}/{data_list[i]}_all.png')
        plt.close()

        # inverse scaling
        train_dataset = dataloader.loader(args.traindir,data_list[i],
                                                seq_size = type_dict[key],loader_type = 'test')

        min_val, max_val = train_dataset.get_minmax()

        inverse_trues = inverse(trues, min_val, max_val)
        inverse_preds = inverse(pred, min_val, max_val)

        plt.figure(figsize=(64, 16))
        plt.plot(trues, color = 'blue',alpha = 0.5, label = 'input_raw')
        plt.plot(pred, color = 'red',alpha = 0.5 ,label = 'output_inverse')
        plt.legend(['input', 'output'])
        plt.savefig(f'{save_path}/{data_list[i]}_all_inverse.png')
        plt.close()

        save_output = pd.DataFrame({'data' : trues,'pred' :pred, 'inverse_data' :inverse_trues, 'inverse_pred' : inverse_preds})

        save_output['loss'] = in_losses

        save_output.to_csv(f'{save_path}/output.csv')
