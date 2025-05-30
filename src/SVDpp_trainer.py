import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
import copy
import os
import time

from .torch_models import *

class SVDppTrainer:
    """
    add some docstring here
    """
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 criterion: nn.Module, # required to be the sum (not mean)
                 reg_lambda: float,
                 device: torch.device,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 scheduler: optim.lr_scheduler = None,
                 rating_range = (1.0, 5.0),
                 early_stopping_patience: int = 5,
                 min_delta: float = 0.0001,
                 verbose=True):

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.reg_lambda = reg_lambda
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.scheduler = scheduler
        self.rating_range = rating_range
        self.rating_min, self.rating_max = rating_range
        self.verbose = verbose
        # early stopping parameters
        self.early_stopping_patience = early_stopping_patience
        self.min_delta = min_delta
        self.epochs_no_improve = 0

        self.best_model_state = None
        self.best_valid_rmse = float('inf')
        self.best_valid_rmse_epoch = 0

    def _train_epoch(self) -> float:
        """train single epoch"""
        self.model.train()

        total_mse_loss = 0.0
        num_samples = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False, disable= not self.verbose)
        for sid_batch, pid_batch, rating_batch, implicit_pids_batch, implicit_lengths_batch in pbar:
            # send data to device
            sid_batch = sid_batch.to(self.device)
            pid_batch = pid_batch.to(self.device)
            rating_batch = rating_batch.to(self.device)
            implicit_pids_batch = implicit_pids_batch.to(self.device)
            implicit_lengths_batch = implicit_lengths_batch.to(self.device)

            batch_size = sid_batch.size(0)

            # zero gradients
            self.optimizer.zero_grad()
            # forward
            predictions = self.model(SIDs=sid_batch, 
                                     PIDs=pid_batch, 
                                     implicit_PIDs=implicit_pids_batch, 
                                     implicit_lengths=implicit_lengths_batch)
            # calculate loss
            mse_loss = self.criterion(predictions, rating_batch)
            reg_loss = self.model.get_l2_reg_loss()
            # combine losses
            total_loss = mse_loss + self.reg_lambda * reg_loss
            # backward
            total_loss.backward()
            # update weights
            self.optimizer.step()
            total_mse_loss += mse_loss.item()
            num_samples += batch_size
            # update progress bar
            batch_rmse = math.sqrt(mse_loss.item() / batch_size) if batch_size > 0 else 0
            pbar.set_postfix({'batch_rmse': f'{batch_rmse:.4f}'})

        # return epoch RMSE
        avg_mse = total_mse_loss / num_samples if num_samples > 0 else 0
        epoch_rmse = math.sqrt(avg_mse)
        return epoch_rmse

    def _evaluate(self) -> float:
        """eval on validation set"""
        self.model.eval()
        total_mse = 0.0
        num_samples = 0

        with torch.no_grad():
            pbar = tqdm(self.valid_loader, desc="Evaluating", leave=False, disable= not self.verbose)
            for sid_batch, pid_batch, rating_batch, implicit_pids_batch, implicit_lengths_batch in pbar:
                # send data to device
                sid_batch = sid_batch.to(self.device)
                pid_batch = pid_batch.to(self.device)
                rating_batch = rating_batch.to(self.device)
                implicit_pids_batch = implicit_pids_batch.to(self.device)
                implicit_lengths_batch = implicit_lengths_batch.to(self.device)

                batch_size = sid_batch.size(0)

                # forward
                predictions = self.model(sid_batch, pid_batch, implicit_pids_batch, implicit_lengths_batch)
                # clip to rating range
                predictions = torch.clamp(predictions, self.rating_min, self.rating_max)
                # calculate loss
                mse = F.mse_loss(predictions, rating_batch, reduction='sum')
                total_mse += mse.item()
                num_samples += batch_size
                batch_rmse = math.sqrt(mse.item() / batch_size) if batch_size > 0 else 0
                pbar.set_postfix({'batch_rmse': f'{batch_rmse:.4f}'})

        # overall RMSE
        avg_mse = total_mse / num_samples if num_samples > 0 else 0
        rmse = math.sqrt(avg_mse)
        return rmse
    
    def evaluate_on_loader(self, data_loader: DataLoader) -> float:
        """eval on a given data loader"""
        self.model.eval()
        total_mse = 0.0
        num_samples = 0

        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluating on loader", leave=False, disable=not self.verbose)
            for sid_batch, pid_batch, rating_batch, implicit_pids_batch, implicit_lengths_batch in pbar:
                sid_batch = sid_batch.to(self.device)
                pid_batch = pid_batch.to(self.device)
                rating_batch = rating_batch.to(self.device)
                implicit_pids_batch = implicit_pids_batch.to(self.device)
                implicit_lengths_batch = implicit_lengths_batch.to(self.device)

                batch_size = sid_batch.size(0)

                predictions = self.model(SIDs=sid_batch, 
                                         PIDs=pid_batch, 
                                         implicit_PIDs=implicit_pids_batch, 
                                         implicit_lengths=implicit_lengths_batch)
                predictions = torch.clamp(predictions, self.rating_min, self.rating_max)

                mse = F.mse_loss(predictions, rating_batch, reduction='sum')
                total_mse += mse.item()
                num_samples += batch_size
                batch_rmse = math.sqrt(mse.item() / batch_size) if batch_size > 0 else 0
                pbar.set_postfix({'batch_rmse': f'{batch_rmse:.4f}'})

        # overall RMSE
        avg_mse = total_mse / num_samples if num_samples > 0 else 0
        rmse = math.sqrt(avg_mse)
        return rmse

    def train(self, num_epochs: int) -> float:
        """
        train the model

        Parameters
        ----------
        num_epochs : int
            number of epochs to train

        Returns
        -------
        float
            best validation RMSE
        """
        if self.verbose:
            print(f"Starting training on {self.device} for {num_epochs} epochs...")
            print(f"Optimizer: {type(self.optimizer).__name__}, LR: {self.optimizer.defaults.get('lr', 'N/A')}")
            print(f"Regularization Lambda: {self.reg_lambda}")
            if self.early_stopping_patience > 0:
                print(f"Early stopping enabled: patience={self.early_stopping_patience}, min_delta={self.min_delta}")

        # reset best RMSE and early stopping parameters
        self.best_valid_rmse = float('inf')
        self.best_valid_rmse_epoch = 0
        self.best_model_state = None
        self.epochs_no_improve = 0

        for epoch in tqdm(range(num_epochs), desc="Training", disable=self.verbose, leave=False):
            start_time = time.time()
            train_rmse = self._train_epoch()
            valid_rmse = self._evaluate()
            end_time = time.time()

            #  check early stopping condition
            if valid_rmse < self.best_valid_rmse - self.min_delta:
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # save best model state (in memory)
            if valid_rmse < self.best_valid_rmse:
                self.best_valid_rmse = valid_rmse
                self.best_valid_rmse_epoch = epoch
                self.best_model_state = copy.deepcopy(self.model.state_dict())

            if self.verbose:
                print(f"[Epoch {epoch+1}/{num_epochs}] "
                      f"Train RMSE: {train_rmse:.4f}, "
                      f"Validation RMSE: {valid_rmse:.4f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}, "
                      f"Duration: {end_time - start_time:.2f}s")
            
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_rmse)
                else:
                    self.scheduler.step()

            # time to stop
            if self.early_stopping_patience > 0 and self.epochs_no_improve >= self.early_stopping_patience:
                if self.verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement for {self.early_stopping_patience} epochs.")
                break

        # load the best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if self.verbose:
                print(f"Loaded model of epoch {self.best_valid_rmse_epoch+1} with best validation RMSE: {self.best_valid_rmse:.4f}")
        else:
             if self.verbose:
                print("no improvement during training, probably something wrong")

        return self.best_valid_rmse
    
    def save_model(self, filepath: str):
        """
        save te model state dictionary to a file

        not tested
        """
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        torch.save(self.model.state_dict(), filepath)
        if self.verbose:
            print(f"Model state saved to {filepath}")