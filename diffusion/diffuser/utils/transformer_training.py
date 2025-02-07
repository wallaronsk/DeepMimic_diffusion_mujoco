import os
import copy
import torch
import numpy as np
from .timer import Timer

def cycle(dl):
    while True:
        for data in dl:
            yield data

class TransformerTrainer:
    def __init__(
        self,
        diffusion_model,
        dataset,
        train_batch_size=32,
        train_lr=1e-4,
        warmup_steps=1000,
        log_freq=100,
        save_freq=1000,
        results_folder='./results',
    ):
        self.model = diffusion_model
        self.dataset = dataset
        self.batch_size = train_batch_size
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.results_folder = results_folder
        
        # Create dataloader
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            num_workers=4,
            shuffle=True,
            pin_memory=True
        ))
        
        # Initialize optimizer with warmup
        self.optimizer = torch.optim.AdamW(
            diffusion_model.parameters(),
            lr=train_lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        
        self.warmup_steps = warmup_steps
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(step / warmup_steps, 1.0)
        )
        
        self.step = 0
        os.makedirs(results_folder, exist_ok=True)

    def train(self, n_train_steps):
        timer = Timer()
        
        for step in range(n_train_steps):
            self.model.train()
            
            # Get batch
            batch = next(self.dataloader)
            
            # Move batch to GPU
            if isinstance(batch, dict):
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            loss, infos = self.model.loss(batch)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            if self.step < self.warmup_steps:
                self.scheduler.step()
            
            # Logging
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
            
            # Saving
            if self.step % self.save_freq == 0:
                self.save()
            
            self.step += 1

    def save(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        savepath = os.path.join(self.results_folder, f'state_{self.step}.pt')
        torch.save(data, savepath)
        print(f'Saved model to {savepath}')

    def load(self, step):
        loadpath = os.path.join(self.results_folder, f'state_{step}.pt')
        data = torch.load(loadpath)
        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.optimizer.load_state_dict(data['optimizer']) 