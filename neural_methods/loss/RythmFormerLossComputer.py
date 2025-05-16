'''
  Adapted from here: https://github.com/ZitongYu/PhysFormer/TorchLossComputer.py
  Modifed based on the HR-CNN here: https://github.com/radimspetlik/hr-cnn
'''
import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from evaluation.post_process import calculate_metric_per_video

def normal_sampling(mean, label_k, std):
    return math.exp(-(label_k-mean)**2/(2*std**2))/(math.sqrt(2*math.pi)*std)

def kl_loss(inputs, labels):
    criterion = nn.KLDivLoss(reduce=False)
    outputs = torch.log(inputs)
    loss = criterion(outputs, labels)
    #loss = loss.sum()/loss.shape[0]
    loss = loss.sum()
    return loss

class Neg_Pearson(nn.Module):    # Pearson range [-1, 1] so if < 0, abs|loss| ; if >0, 1- loss
    def __init__(self):
        super(Neg_Pearson,self).__init__()

    def forward(self, preds, labels):       # all variable operation
        loss = 0
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])                # x
            sum_y = torch.sum(labels[i])               # y
            sum_xy = torch.sum(preds[i]*labels[i])        # xy
            sum_x2 = torch.sum(torch.pow(preds[i],2))  # x^2
            sum_y2 = torch.sum(torch.pow(labels[i],2)) # y^2
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += 1 - pearson
            
        loss = loss/preds.shape[0]
        return loss
    
class Smooth_Neg_Pearson(nn.Module):
    """
    The Neg_Pearson Module is from the orignal author of Physnet.
    Code of 'Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks' 
    source: https://github.com/ZitongYu/PhysNet/blob/master/NegPearsonLoss.py
    """
    def __init__(self, loss_exp=2, window_size=7):
        super(Smooth_Neg_Pearson, self).__init__()
        self.loss_exp = loss_exp
        self.window_size = window_size

    def smooth(self, x):
        '''input is a batched 1-D signal [N, T]'''
        kernel = torch.ones(self.window_size) / self.window_size
        kernel = kernel.to(x.device)
        kernel = kernel.view(1, 1, -1)
        x = x.unsqueeze(1) # Add a channel dimension
        x = torch.nn.functional.conv1d(x, kernel, padding=self.window_size//2)
        return x.squeeze(1)

    def forward(self, preds, labels): 
        #print('preds', preds.shape, 'labels', labels.shape)      
        loss = 0
        preds = self.smooth(preds)
        labels = self.smooth(labels)
        for i in range(preds.shape[0]):
            sum_x = torch.sum(preds[i])               
            sum_y = torch.sum(labels[i])             
            sum_xy = torch.sum(preds[i]*labels[i])       
            sum_x2 = torch.sum(torch.pow(preds[i],2))  
            sum_y2 = torch.sum(torch.pow(labels[i],2)) 
            N = preds.shape[1]
            pearson = (N*sum_xy - sum_x*sum_y)/(torch.sqrt((N*sum_x2 - torch.pow(sum_x,2))*(N*sum_y2 - torch.pow(sum_y,2))))
            loss += (1 - pearson) ** self.loss_exp
        loss = loss/preds.shape[0]
        return loss

class RhythmFormer_Loss(nn.Module): 
    def __init__(self):
        super(RhythmFormer_Loss,self).__init__()
        self.criterion_Pearson = Smooth_Neg_Pearson()
    def forward(self, pred_ppg, labels , epoch , FS , diff_flag):   
        loss_time = self.criterion_Pearson(pred_ppg.view(1,-1) , labels.view(1,-1))
        #print('1', loss_time)    
        loss_CE , loss_distribution_kl = TorchLossComputer.Frequency_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        #print('2', loss_CE, loss_distribution_kl)
        loss_hr = TorchLossComputer.HR_loss(pred_ppg.squeeze(-1),  labels.squeeze(-1), diff_flag=diff_flag, Fs=FS, std=3.0)
        #print( 'hr_loss output', loss_hr)
        if torch.isnan(loss_time) : 
           loss_time = 0
        loss = 0.2 * loss_time + 1.0 * loss_CE + 1.0 * loss_hr
        return loss

class TorchLossComputer(object):
    @staticmethod
    def compute_complex_absolute_given_k(output, k, N):
        two_pi_n_over_N = Variable(2 * math.pi * torch.arange(0, N, dtype=torch.float), requires_grad=True) / N
        hanning = Variable(torch.from_numpy(np.hanning(N)).type(torch.FloatTensor), requires_grad=True).view(1, -1)

        k = k.type(torch.FloatTensor).cuda()
        two_pi_n_over_N = two_pi_n_over_N.cuda()
        hanning = hanning.cuda()
            
        output = output.view(1, -1) * hanning
        output = output.view(1, 1, -1).type(torch.cuda.FloatTensor)
        k = k.view(1, -1, 1)
        two_pi_n_over_N = two_pi_n_over_N.view(1, 1, -1)
        complex_absolute = torch.sum(output * torch.sin(k * two_pi_n_over_N), dim=-1) ** 2 \
                           + torch.sum(output * torch.cos(k * two_pi_n_over_N), dim=-1) ** 2

        return complex_absolute

    @staticmethod
    def complex_absolute(output, Fs, bpm_range=None):
        output = output.view(1, -1)

        N = output.size()[1]

        unit_per_hz = Fs / N
        feasible_bpm = bpm_range / 60.0
        k = feasible_bpm / unit_per_hz

        # only calculate feasible PSD range [0.7,4]Hz
        complex_absolute = TorchLossComputer.compute_complex_absolute_given_k(output, k, N)

        return (1.0 / complex_absolute.sum()) * complex_absolute	# Analogous Softmax operator


    @staticmethod
    def cross_entropy_power_spectrum_loss(inputs, target, Fs):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)

    @staticmethod
    def cross_entropy_power_spectrum_focal_loss(inputs, target, Fs, gamma):
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)
        
        #pdb.set_trace()
        criterion = FocalLoss(gamma=gamma)

        #return F.cross_entropy(complex_absolute, target.view((1)).type(torch.long)).view(1),  (target.item() - whole_max_idx.item()) ** 2
        return criterion(complex_absolute, target.view((1)).type(torch.long)),  torch.abs(target[0] - whole_max_idx)


    @staticmethod
    def cross_entropy_power_spectrum_forward_pred(inputs, Fs):
        inputs = inputs.view(1, -1)
        bpm_range = torch.arange(40, 190, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 180, dtype=torch.float).cuda()
        #bpm_range = torch.arange(40, 260, dtype=torch.float).cuda()

        complex_absolute = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)

        whole_max_val, whole_max_idx = complex_absolute.view(-1).max(0)
        whole_max_idx = whole_max_idx.type(torch.float)

        return whole_max_idx

    @staticmethod
    def Frequency_loss(inputs, target, diff_flag , Fs, std):
        hr_gt, pred_hr_peak, SNR, macc = calculate_metric_per_video(inputs.detach().cpu(), target.detach().cpu(), diff_flag = diff_flag, fs=Fs, hr_method='FFT')
         
        inputs = inputs.view(1, -1)
        target = target.view(1, -1)
        bpm_range = torch.arange(36, 198, dtype=torch.float).to(torch.device('cuda'))
        ca = TorchLossComputer.complex_absolute(inputs, Fs, bpm_range)
        sa = ca/torch.sum(ca)

        target_distribution = [normal_sampling(int(hr_gt), i, std) for i in range(36, 198)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))

        hr_gt = torch.tensor(hr_gt-36).view(1).type(torch.long).to(torch.device('cuda'))
    
        return F.cross_entropy(ca, hr_gt) , kl_loss(sa , target_distribution)

    @staticmethod
    def HR_loss(inputs, target,  diff_flag , Fs, std):
        psd_gt, psd_pred, SNR, macc = calculate_metric_per_video(inputs.detach().cpu(), target.detach().cpu(), diff_flag = diff_flag, fs=Fs, hr_method='Peak')
        
        if psd_pred < 36 or psd_pred > 198:
            psd_pred = np.clip(psd_pred, 36, 198)
            print('clipped pred')
        pred_distribution = [normal_sampling(psd_pred, i, std) for i in range(36,198)]
        pred_distribution = [i if i > 1e-15 else 1e-15 for i in pred_distribution]
        #print(pred_distribution)
        pred_distribution = torch.Tensor(pred_distribution).to(torch.device('cuda'))
        
        if psd_gt < 36 or psd_gt > 198:
            psd_gt = np.clip(psd_gt, 36, 198)
            print('clipped gt')
        target_distribution = [normal_sampling(psd_gt, i, std) for i in range(36,198)]
        target_distribution = [i if i > 1e-15 else 1e-15 for i in target_distribution]
        #print(target_distribution)
        target_distribution = torch.Tensor(target_distribution).to(torch.device('cuda'))
        return kl_loss(pred_distribution , target_distribution)
