"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Smooth_Neg_Pearson
from neural_methods.loss.ContrastLoss import ContrastLoss, CalculateNormPSD
from neural_methods.loss.SNRLoss import SNRLoss_dB_Signals
from neural_methods.model.ContrastFusion import ContrastFusion
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm


class ContrastFusionTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        self.model = ContrastFusion(S=config.MODEL.CONTRASTFUSION.S, 
                                  in_ch=config.MODEL.CONTRASTFUSION.CHANNELS).to(self.device)        
        if config.MODEL.PRETRAINED is not None:
            self.model.load_state_dict(torch.load(config.MODEL.PRETRAINED, 
                                                  map_location=self.device))
            print("Pre-trained:", config.MODEL.PRETRAINED)
        else:
            print("No pre-trained model loaded!")
        if self.num_of_gpu > 0:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))        

        if config.MODEL.TYPE == "RR":
            self.lower_cutoff = 5
            self.upper_cutoff = 45
            window_size = 15
        elif config.MODEL.TYPE == "HR":
            self.lower_cutoff = 40
            self.upper_cutoff = 250
            window_size = 7
        else:
            raise ValueError("Model type not supported!")

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.contrast_loss = ContrastLoss(config.MODEL.CONTRASTFUSION.FRAME_NUM, 1, 
                                              config.TRAIN.DATA.FS, self.lower_cutoff, self.upper_cutoff,
                                              dist='combined')
            print(self.contrast_loss.distance_func)
            self.loss_model = Smooth_Neg_Pearson(2, window_size)
            self.snr_loss = SNRLoss_dB_Signals(pulse_band = [self.lower_cutoff / 60, self.upper_cutoff / 60], 
                                               Fs=config.TRAIN.DATA.FS)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=config.TRAIN.LR)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        norm = CalculateNormPSD(self.config.TRAIN.DATA.FS, self.lower_cutoff, self.upper_cutoff)
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                model_out = self.model(batch[0].to(torch.float32).to(self.device))
                rPPG = model_out[:, -1]
                BVP_label = batch[1].to(torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG, dim=1, keepdim=True)) \
                            / torch.std(rPPG, dim=1, keepdim=True)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label, dim=1, keepdim=True)) \
                            / torch.std(BVP_label, dim=1, keepdim=True)  # normalize
                
                loss, p_loss, n_loss, p_loss_gt, n_loss_gt = self.contrast_loss(model_out, BVP_label,
                                                                                torch.ones(2).to(self.device))
                loss += self.loss_model(rPPG, BVP_label) + self.snr_loss(rPPG, BVP_label)
                loss.backward()

                try:
                    with torch.no_grad():
                        rppg_psd = [norm(rPPG[i]) for i in range(rPPG.shape[0])]
                        bvp_psd = [norm(BVP_label[i]) for i in range(BVP_label.shape[0])]

                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(rPPG[0].detach().cpu().numpy(), label='rPPG')
                    plt.plot(BVP_label[0].detach().cpu().numpy(), label='BVP')
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(rPPG[1].detach().cpu().numpy(), label='rPPG')
                    plt.plot(BVP_label[1].detach().cpu().numpy(), label='BVP')
                    plt.legend()
                    plt.savefig("rPPG_BVP.png")
                    plt.close()

                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.plot(rppg_psd[0].detach().cpu().numpy(), label='rPPG PSD')
                    plt.plot(bvp_psd[0].detach().cpu().numpy(), label='BVP PSD')
                    plt.legend()
                    plt.subplot(1, 2, 2)
                    plt.plot(rppg_psd[1].detach().cpu().numpy(), label='rPPG PSD')
                    plt.plot(bvp_psd[1].detach().cpu().numpy(), label='BVP PSD')
                    plt.legend()
                    plt.savefig("rPPG_BVP_PSD.png")
                    plt.close()
                except Exception as e:
                    print("Error in plotting PSD: ", e)
                    print("Check Shapes")
                    print(rPPG.shape, BVP_label.shape, batch[0].shape)

                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Append the current learning rate to the list
                lrs.append(0)

                self.optimizer.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH: 
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH: 
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)
                rPPG = self.model(valid_batch[0].to(torch.float32).to(self.device))[:,-1]
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, _, _, _ = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs 
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
