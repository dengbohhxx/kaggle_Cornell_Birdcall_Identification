# -*- coding: utf-8 -*-
import os
import torch
import datetime
import time
from utlis.averagemeter import AverageMeter
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from utlis.row_wise_micro_averaged_f1_score import row_wise_micro_averaged_f1_score
class Fitter:
    def __init__(self, model, device, config):       
        self.config = config      
        self.epoch = 0  
        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)         
        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        self.best_micro_average_f1_score=-1
        self.model = model
        self.device = device        
        param_optimizer = list(self.model.named_parameters())        
        no_decay = ['bias', 'bn']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay )],'weight_decay': 0.0},
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay': 0.001}
        ] 
        self.optimizer = torch.optim.RMSprop(optimizer_grouped_parameters, lr=config.lr,momentum=0.9)
        self.scheduler =config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')
        self.writer=SummaryWriter('output/tensorboard')
    def fit(self, train_loader, validation_loader):
        if self.config.verbose:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('lr',lr)
            timestamp = datetime.datetime.now().utcnow().isoformat()
            self.log(f'\n{timestamp}\nLR: {lr}')           
        t = time.time()
        summary_loss = self.train_one_epoch(train_loader)
        self.log(
            f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
        self.save(f'{self.base_dir}/last-checkpoint.tar')
        t = time.time()
        summary_loss,micro_average_f1_score = self.validation(validation_loader)
        self.writer.add_scalar('f1_score',micro_average_f1_score.avg)
        self.log(
            f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f},micro_average_f1_score:{micro_average_f1_score.avg:.5f},time: {(time.time() - t):.5f}')
        if micro_average_f1_score.avg > self.best_micro_average_f1_score:
            self.best_micro_average_f1_score = micro_average_f1_score.avg
            self.model.eval()
            self.save(f'{self.base_dir}/best-f1score-checkpoint-{str(self.epoch).zfill(3)}epoch.tar')
        if summary_loss.avg < self.best_summary_loss:
            self.best_summary_loss = summary_loss.avg
        if self.config.validation_scheduler:
            self.scheduler.step(metrics=summary_loss.avg)
        self.epoch += 1    
    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        micro_average_f1_score=AverageMeter()
        t = time.time()
        for step, (waveform,labels) in enumerate(val_loader):
            with torch.no_grad(): 
                batch_size=self.config.batch_size
                #print(waveform)
                waveform=torch.tensor(waveform).to(self.device).float()
                labels=torch.tensor(labels).to(self.device).float()
                loss_v,pred= self.model(waveform,labels)
                print(pred)
                self.writer.add_scalar('VAL_LOSS', loss_v,batch_size*(step+1))
                summary_loss.update(loss_v.detach().item(), batch_size)
                f1_score=row_wise_micro_averaged_f1_score(pred,labels,0.5)
                micro_average_f1_score.update(f1_score, batch_size)
                if self.config.verbose:
                    if step % self.config.verbose_step == 0:
                        print(
                            f'Val Step {step}/{len(val_loader)}, ' + \
                                f'summary_loss: {summary_loss.avg:.5f}, ' + \
                                    f'micro_average_f1_score: {micro_average_f1_score.avg:.5f}'+\
                                        f'time: {(time.time() - t):.5f}', end='\r'
                       )
        return summary_loss,micro_average_f1_score

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (waveform,labels) in enumerate(train_loader):
            if step==1:
                break
            batch_size=self.config.batch_size
            self.optimizer.zero_grad()                       
            labels=torch.tensor(labels).to(self.device).float()
            waveform=torch.tensor(waveform).to(self.device).float()
            loss_t,_ = self.model(waveform,labels)
            if step+self.epoch==0:
                self.writer.add_graph(self.model,(waveform,labels))
            self.writer.add_scalar('TRAIN_LOSS',loss_t,batch_size*(step+1))
            loss_t.backward()
            summary_loss.update(loss_t.detach().item(), batch_size)
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    ) 
            self.optimizer.step() 
            if self.config.step_scheduler:
                self.scheduler.step()
        return summary_loss  

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
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