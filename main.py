from models.Informer.model import Informer, InformerStack
from models.Linformer.model import Linformer
from torchsummaryX import summary as summaryx
from torchsummary import summary
# from models.Reformer.reformer_enc_dec import ReformerEncDec
from models.LeNet5.lenet import LeNet5
from models.Seq2Seq.seq2seq import Seq2SeqEncDec
import dataloader
import utils.utils as utils
from utils.metrics import MSE,MAE,RMSE
# from utils.tools import EarlyStopping, adjust_learning_rate
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
# from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['agg.path.chunksize'] = 10000
parser = argparse.ArgumentParser(description='Time series forecasting')

parser.add_argument('--model', type=str, default='informer',help='model of experiment, options: [informer, informerstack, informerlight(TBD), linformer]')

parser.add_argument('--dataset', type=str, help='dataset: e, stock', default = None)
parser.add_argument('--folder_name', type=str, help='exp-folder-name', default = '')

parser.add_argument('--epochs', default=200, type=int, help='epoch (default: 200)')
parser.add_argument('--batch_size', default=1500, type=int, help='batch size (default: 1024)')
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
parser.add_argument('--padding', type=int, default=0, help='padding type, -1 = use target sequence as decoder input')
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


## Reformer options

parser.add_argument('--enc_depth', type=int, default = 6, help ='Reformer/ model depth')
parser.add_argument('--enc_bucket_size',type=int, default = 8 ,help ='Reformer/ bucket size')
parser.add_argument('--enc_n_hashes', type=int, default = 4, help ='Reformer/ the number of hashes ')
parser.add_argument('--enc_ff_chunks',type=int, default = 10, help='Reformer/ ff_chunks')
parser.add_argument('--enc_lsh_dropout',type=float, default = 0.1,help ='Reformer/ lsh layer dropout')
parser.add_argument('--enc_weight_tie',action='store_true', help = 'Reformer/ maybe query & key weigth ties', default = True)
parser.add_argument('--enc_causal', action= 'store_true', help = 'Reformer/ causal', default = True)
parser.add_argument('--enc_n_local_attn_heads', type = int, help= 'Reformer / n_local_attn_heads', default = 4)
parser.add_argument('--enc_use_full_attn', action= 'store_true', help = 'Reformer/use_full_attn', default  = False)
parser.add_argument('--enc_heads', type=int, default = 8, help ='Reformer/enc_heads ')
parser.add_argument('--enc_dim_head',type=int, default = 32, help='Reformer/enc dim_head')
parser.add_argument('--enc_attn_chunks',type=int, default = 1,help ='Reformer/enc attn_chunks')

parser.add_argument('--dec_depth', type=int, default = 6, help ='Reformer/ model depth')
parser.add_argument('--dec_bucket_size',type=int, default = 8 ,help ='Reformer/ bucket size')
parser.add_argument('--dec_n_hashes', type=int, default = 4, help ='Reformer/ the number of hashes ')
parser.add_argument('--dec_ff_chunks',type=int, default = 10, help='Reformer/ ff_chunks')
parser.add_argument('--dec_dropout',type=float, default = 0.1,help ='Reformer Seq2Seq/ lsh layer dropout')
parser.add_argument('--dec_weight_tie',action='store_true', help = 'Reformer/ maybe query & key weigth ties', default = True)
parser.add_argument('--dec_causal', action= 'store_true', help = 'Reformer/ causal', default = True)
parser.add_argument('--dec_n_local_attn_heads', type = int, help= 'Reformer / n_local_attn_heads', default = 4)
parser.add_argument('--dec_use_full_attn', action= 'store_true', help = 'Reformer/use_full_attn', default  = False)
parser.add_argument('--dec_heads', type=int, default = 8, help ='Reformer/dec_heads ')
parser.add_argument('--dec_dim_head',type=int, default = 32, help='Reformer/ dec dim_head')
parser.add_argument('--dec_attn_chunks',type=int, default = 1,help ='Reformer/ dec attn_chunks')
                        
## Seq2Seq options
parser.add_argument('--teacher_force', type = float, help = 'Seq2Seq/teacher_force prob', default  = 0)
parser.add_argument('--rnn_num_layers',type = int, help = 'Seq2Seq/rnn_num_layers', default  =  1)
parser.add_argument('--enc_dropout',type = float, help = 'Seq2Seq/enc_dropout', default  =  0.3)
                        
## Linformer options
parser.add_argument('--enc_k', type = int, help = 'Linformer/ K ', default = None)
parser.add_argument('--dec_k', type = int, help = 'Linformer/ K ', default = None)
parser.add_argument('--headwise_sharing', action= 'store_true', help = 'Linformer/use headwise sharing', default  = False)
parser.add_argument('--key_value_sharing', action= 'store_true', help = 'Linformer/use key value sharing', default  = False)

parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

args = parser.parse_args()

def inverse(x, mini, maxi):
    output = mini + x*(maxi - mini)
    return output

def train(loader, net, criterion, optimizer, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.train()
    for i, (input,target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        input = input.float().cuda()
        target = target.float().cuda()
        
        if args.padding==0:
            dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
            dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
        elif args.padding==1:
            dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
            dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
        elif args.padding == -1:
            dec_inp = target.float().cuda()
              
#         print(args.output_attention)
        if args.output_attention:
            outputs, attens = net(input, dec_inp)
            target = target[:,-args.pred_len:,0:].cuda()
        elif args.model == 'seq2seq':
            output = net(input, target)
        else:
            outputs = net(input)
            
        loss = criterion(outputs, target)
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[train] Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    logger.write([epoch, losses.avg,batch_time.avg])
   
    return epoch, losses.avg

def test(loader, net, criterion, epoch, logger, args):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.eval()
    preds = []
    trues = []
    for i, (input,target) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.float().cuda()
        target = target.float().cuda()
        
        if args.padding==0:
            dec_inp = torch.zeros([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
            dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
        elif args.padding==1:
            dec_inp = torch.ones([target.shape[0], args.pred_len, target.shape[-1]]).float().cuda()
            dec_inp = torch.cat([target[:,:args.label_len,:], dec_inp], dim=1).float().cuda()
        elif args.padding == -1:
            dec_inp = target.float().cuda()
        
        
        if args.output_attention:
            outputs, attens = net(input, dec_inp)
            target = target[:,-args.pred_len:,0:].cuda()
        elif args.model == 'seq2seq':
            output = net(input, target)
        else:
            outputs = net(input)
        
        loss = criterion(outputs.detach().cpu(), target.detach().cpu())
        
        losses.update(loss.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('[test] Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

    logger.write([epoch, losses.avg,batch_time.avg])
    
   
    return epoch, losses.avg, preds, trues

def evaluate(loader, net, criterion):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    end = time.time()
    net.eval()
    preds = []
    trues = []
    with torch.no_grad():
            for i, (inputs,target) in enumerate(loader):
                args.batch = inputs.shape[0]
                
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
                elif args.model == 'seq2seq':
                    output = net(input, target)
                else:
                    outputs = net(input)
                if i < 6 :
                    plt.figure(figsize=(64, 16))
                    plt.plot(target.reshape(-1,1).detach().cpu().numpy()[:1000], color = 'blue',alpha = 0.5, linewidth = 3,label = 'input')
                    plt.plot(outputs.reshape(-1,1).detach().cpu().numpy()[:1000], color = 'red',alpha = 0.5 , linewidth = 3,label = 'output')
                    plt.legend(['target', 'prediction'], prop={'size': 30})
                    plt.savefig(f'{args.save_path}/pred_{i}.png')
                    plt.close()
                
                preds = np.append(preds,outputs[:,0,:].detach().cpu().numpy())
                trues = np.append(trues,target[:,0,:].detach().cpu().numpy())

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

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    file_list = sorted(os.listdir(args.traindir))
    data_list = [file for file in file_list if file.endswith(".csv")]
    if args.dataset == 'e':
        data_list = ['e_DOM.csv', 'e_AEP.csv']
    elif args.dataset == 'stock':
        data_list = ['AMD.csv' ,'NVDA.csv']
    
    for i in range(len(data_list)):
        if data_list[i] == 'AMD.csv' or data_list[i] == 'NVDA.csv':
            type_dict = {
                    'type0': [7, 5, 1],
                    'type1': [35, 14, 7],
                    'type2': [70, 30, 14],
                    'type3': [100, 50, 30],
                    # 'type4': [100, 50, 30],
                }
        else :
            type_dict = {
                    'type0': [7, 5, 1],
                    'type1': [168, 72, 24],
                    'type2': [300, 100, 1],
                    'type3': [300, 100, 50],
                    # 'type4': [450, 200, 100],
                }
        for key in type_dict.keys() : 

            args.seq_len = type_dict[key][0]
            args.label_len = type_dict[key][1]
            args.pred_len = type_dict[key][2]


            args.save_root = f'./exp/{args.model}-{key}-{args.folder_name}/'

            train_dataset = dataloader.loader(args.traindir,data_list[i],
                                            seq_size = type_dict[key],loader_type = 'train',args = args)
            
            test_dataset = dataloader.loader(args.traindir,data_list[i],
                                            seq_size = type_dict[key],loader_type = 'test',args = args)


            save_path = os.path.join(args.save_root,data_list[i])
            
            args.save_path = save_path
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            train_loader = DataLoader(train_dataset,
                                        shuffle=True, 
                                        batch_size=args.batch_size, 
                                        pin_memory=False)
            test_loader = DataLoader(test_dataset,
                                    shuffle=False, 
                                    batch_size=args.batch_size, 
                                    pin_memory=False)


            model_dict = {
                'informer':Informer,
                'informerstack':InformerStack,
                #  'reformer':ReformerEncDec,
                'LeNet5' : LeNet5,
                'lstm' : LSTM,
                'seq2seq': Seq2SeqEncDec,
                'linformer': Linformer 
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

                # input_data = torch.randn(1, 100,1).cuda()
                # other_input_data = torch.randn(1, 300,1).cuda()

                # summaryx(net,input_data, other_input_data)
                # import ipdb; ipdb.set_trace()


            elif args.model == 'LeNet5':
                net = model_dict[args.model](
                        seq_len = args.seq_len,
                        pred_len=args.pred_len
                ).float().cuda()

                # summary(net,(300,1))
                # import ipdb; ipdb.set_trace()

            elif args.model == 'lstm':
                net = model_dict[args.model](
                        n_hidden = 256,
                        seq_len = args.seq_len,
                        pred_len=args.pred_len
                ).float().cuda()
                # input_data = torch.randn(1,300,1).cuda()
                # summaryx(net,input_data)
                # import ipdb; ipdb.set_trace()
            
            elif args.model == 'seq2seq':
                net = model_dict[args.model](
                        input_len = args.seq_len,
                        label_len = args.label_len ,
                        pred_len = args.pred_len ,
                        input_dim = args.enc_in,
                        hidden_size = args.hidden_size,
                        rnn_num_layers = args.rnn_num_layers,
                        teacher_force = args.teacher_force,
                        enc_dropout = args.enc_dropout,
                        dec_dropout = args.dec_dropout, 
                        ).cuda()
                # input_data = torch.randn(1,300,1).cuda()
                # summaryx(net,input_data)
                # import ipdb; ipdb.set_trace()
            elif args.model=='linformer' :
                e_layers = args.e_layers if args.model=='linformer' else args.s_layers ################################
                enc_k = (args.seq_len)//2   if ( args.enc_k is None)  else args.enc_k
                dec_k = (args.pred_len + args.label_len )//2 if (args.dec_k is None) else args.dec_k
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
                    activation =args.activation,
                    output_attention = args.output_attention,
                    distil = args.distil,
                    mix = args.mix,
                    enc_k = enc_k,
                    dec_k = dec_k,
                    headwise_sharing = args.headwise_sharing , key_value_sharing = args.key_value_sharing
                ).float().cuda()

            
            elif args.model == 'reformer':
                enc_bucket_size = args.seq_len//2 if args.enc_bucket_size == 0 else args.enc_bucket_size
                dec_bucket_size = (args.label_len+ args.pred_len)//2 if args.enc_bucket_size== 0 else args.enc_bucket_size
                net = model_dict[args.model]( dim= args.enc_in, seq_len= args.seq_len,
                        label_len = args.label_len,
                        pred_len = args.pred_len,
                        enc_bucket_size =  enc_bucket_size, # default: maxlen 128 , bucket_size 64
                        dec_bucket_size = dec_bucket_size,# default: maxlen 128 , bucket_size 64
                        enc_depth = args.enc_depth,
                         dec_depth = args.dec_depth,
                        enc_heads =args.enc_heads,  #default: 8
                        dec_heads = args.dec_heads,  #default: 8
                        enc_dim_head = args.enc_dim_head, 
                        dec_dim_head= args.dec_dim_head, 
                        enc_n_hashes=args.enc_n_hashes,
                        dec_n_hashes=args.dec_n_hashes,
                        enc_ff_chunks= args.enc_ff_chunks,
                        dec_ff_chunks= args.dec_ff_chunks,
                        enc_attn_chunks= args.enc_attn_chunks,
                        dec_attn_chunks= args.dec_attn_chunks,
                        enc_weight_tie= args.enc_weight_tie,
                        dec_weight_tie= args.dec_weight_tie,
                        enc_causal= args.enc_causal,
                        dec_causal= args.dec_causal,
                        enc_n_local_attn_heads= args.enc_n_local_attn_heads,
                        dec_n_local_attn_heads= args.dec_n_local_attn_heads,
                        enc_use_full_attn=args.enc_use_full_attn, 
                        dec_use_full_attn= args.dec_use_full_attn).cuda()


            # net = nn.DataParallel(net)

            criterion = nn.MSELoss().cuda()

            optimizer = optim.Adam(net.parameters(),lr=1e-3)

            train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
            test_logger = utils.Logger(os.path.join(save_path, 'test.log'))

            # Start Train
            for epoch in range(1, args.epochs+1):
                epoch,loss = train(train_loader,
                                    net,
                                    criterion,
                                    optimizer,
                                    epoch,
                                    train_logger,
                                    args)
                epoch, tst_loss, preds, trues = test(test_loader, net, criterion, epoch, test_logger, args)
            
            pred, trues = evaluate(test_loader, net, criterion)

            torch.save(net.state_dict(),
                    os.path.join(save_path, f'model_{int(args.epochs)}.pth'))

            mse_losses = MSE(pred, trues)
            mae_losses = MAE(pred, trues)
            rmse_losses = RMSE(pred, trues)
            
            plt.figure(figsize=(64, 16))
            plt.plot(trues, color = 'blue',alpha = 0.5, linewidth = 3,label = 'input')
            plt.plot(pred, color = 'red',alpha = 0.5 ,linewidth = 3,label = 'output')
            plt.legend(['target', 'prediction'], prop={'size': 30})
            plt.savefig(f'{save_path}/{data_list[i]}_all.png')
            plt.close()

            # inverse scaling

            min_val, max_val = train_dataset.get_minmax()

            inverse_trues = inverse(trues, min_val, max_val)
            inverse_preds = inverse(pred, min_val, max_val)

            plt.figure(figsize=(64, 16))
            plt.plot(inverse_trues, color = 'blue',alpha = 0.5,linewidth = 3, label = 'input_raw')
            plt.plot(inverse_preds, color = 'red',alpha = 0.5 ,linewidth = 3,label = 'output_inverse')
            plt.legend(['target', 'prediction'], prop={'size': 30})
            plt.savefig(f'{save_path}/{data_list[i]}_all_inverse.png')
            plt.close()

            save_output = pd.DataFrame({'data' : trues,'pred' :pred, 'inverse_data' :inverse_trues, 'inverse_pred' : inverse_preds})

            save_output['MSE'] = mse_losses
            save_output['MAE'] = mae_losses
            save_output['RMSE'] = rmse_losses

            save_output.to_csv(f'{save_path}/output.csv')


if __name__ == "__main__":
    main()