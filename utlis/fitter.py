# -*- coding: utf-8 -*-
import os
import torch
import datetime
import time
from utlis.averagemeter import AverageMeter
from utlis.pytorch_utils import do_mixup
from models.Loss import Contrastive_loss
from torch.utils.tensorboard import SummaryWriter
class Fitter:
    def __init__(self, model, device, config):       
        self.config = config      
        self.epoch = 0  
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)         
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        self.model = model
        self.device = device        
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'bn']
        no_lr_no_decay=['logmel_extractor','spectrogram_extractor']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_lr_no_decay)], 'lr':0.0},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_lr_no_decay) and any(nd in n for nd in no_decay)]},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_lr_no_decay) and not any(nd in n for nd in no_decay)]}
        ]
        self.optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=config.lr)
        self.scheduler =config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        self.writer=SummaryWriter('output/tensorboard')
        self.train_steps = 0
        self.val_steps = 0

        self.mixup = True
        self.pre_x = torch.ones((config.batch_size, 160000)).cuda()
        self.pre_y = torch.ones((config.batch_size, 264)).cuda()
        self.clr_loss = Contrastive_loss(config.batch_size)
        
    def fit(self, train_loader, validation_loader):
        if self.config.verbose:
            lr = self.optimizer.param_groups[0]['lr']
            timestamp = datetime.datetime.now().utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')           
        t = time.time()
        summary_loss = self.train_one_epoch(train_loader)
        self.log(
            f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
        self.save(f'{self.base_dir}/last-checkpoint.tar')
        t = time.time()
        summary_loss = self.validation(validation_loader)
        self.log(
            f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
        if summary_loss.avg < self.best_summary_loss:
            self.best_summary_loss = summary_loss.avg
            self.model.eval()
            self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.tar')
        if self.config.validation_scheduler:
            self.scheduler.step(metrics=summary_loss.avg)
        self.epoch += 1    
    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (waveform,labels) in enumerate(val_loader):
            with torch.no_grad(): 
                batch_size=self.config.batch_size
                self.val_steps += batch_size
                waveform=torch.tensor(waveform).to(self.device).float()
                labels=torch.tensor(labels).to(self.device).float()
                loss_v, _= self.model(waveform,labels)
                loss_v = loss_v.mean()
                self.writer.add_scalar('VAL_LOSS', loss_v, self.val_steps)
                summary_loss.update(loss_v.detach().item(), batch_size)
                if self.config.verbose:
                    if step % self.config.verbose_step == 0:
                        print(
                            f'Val Step {step}/{len(val_loader)}, ' + \
                                f'summary_loss: {summary_loss.avg:.8f}, ' + \
                                    f'time: {(time.time() - t):.5f}', end='\r'
                       )
        return summary_loss

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (waveform,labels) in enumerate(train_loader):
            batch_size=self.config.batch_size
            self.train_steps += batch_size
            self.optimizer.zero_grad()            
            labels=torch.tensor(labels).to(self.device).float()
            waveform=torch.tensor(waveform).to(self.device).float()
            if self.mixup and (waveform.shape[0] == self.pre_x.shape[0]):
                self.pre_x = self.pre_x[0:waveform.shape[0],:]
                self.pre_y = self.pre_y[0:waveform.shape[0],:]
                x_mixup = do_mixup(waveform.cpu(), self.pre_x.cpu(), mixup_lambda=0.3).cuda()
                #y_mixup = do_mixup(labels.cpu(), self.pre_y.cpu(), mixup_lambda=0.3).cuda()
                loss_t_1, z1 = self.model(x_mixup,labels)
                loss_t_2, _ = self.model(x_mixup,self.pre_y)
                loss_t = do_mixup(loss_t_1.cpu(), loss_t_2.cpu(), mixup_lambda=0.3).cuda()
                # contrastive learning
                _, z_ori = self.model(waveform, labels)
                loss_clr = self.clr_loss(z_ori, z1)
                self.pre_x = waveform.clone()
                self.pre_y = labels.clone()
            else:
                loss_t, _ = self.model(waveform,labels)
            loss_t = loss_t.mean() + loss_clr.mean()
            #if step+self.epoch==0:
            #    self.writer.add_graph(self.model.module, (waveform,labels))
            self.writer.add_scalar('TRAIN_LOSS', loss_t, self.train_steps)
            loss_t.backward()
            summary_loss.update(loss_t.detach().item(), batch_size)
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.8f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    ) 
            self.optimizer.step()
            if self.config.step_scheduler:
                self.scheduler.step()
    
        return summary_loss  

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.module.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1
    
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
