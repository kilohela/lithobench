import sys
sys.path.append(".")
from lithobench.model import *
import pylitho.exact as litho
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from mine.utils import color
from lithobench.dataset import *

class Freq01nn(nn.Module):
    def __init__(self, in_size=2048, out_size=64):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        conv1 = nn.Conv2d(2, 8, kernel_size=3, stride=2, padding=1)
        conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1)
        conv3 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1)
        conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        out_conv1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        out_conv2 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        out_conv3 = nn.Conv2d(32, 8, kernel_size=1, stride=1)
        out_conv4 = nn.Conv2d(8, 2, kernel_size=1, stride=1)

        self._seq = nn.Sequential(conv1, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(8),
                                 conv2, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(32),
                                 conv3, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(128),
                                 conv4, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(256),
                                 conv5, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(512),
                                 out_conv1, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(128),
                                 out_conv2, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(32),
                                 out_conv3, nn.ReLU(inplace=True), torch.nn.BatchNorm2d(8),
                                 out_conv4)

    def forward(self, x):
        return self._seq(x)
    
def cosine_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=0.0):
    def lr_lambda(current_step):
        # --- warmup ---
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))

        # --- cosine ---
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))

        # scale 到 min_lr
        return cosine * (1 - min_lr) + min_lr
    
    return LambdaLR(optimizer, lr_lambda)
    
class Freq01(ModelILT):
    def __init__(self, size=(512, 512), radius=32): 
        super().__init__(size=size, name="Freq01")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = Freq01nn().to(self.device)
        self.radius = radius

        self.model_name = self.__class__.__name__
        self.checkpoints_dir = "./mine/checkpoints"
        self.latest_checkpoint_pretrain = os.path.join(self.checkpoints_dir, self.model_name + "_pretrain_latest.pth")
        self.best_checkpoint_pretrain = os.path.join(self.checkpoints_dir, self.model_name + "_pretrain_best.pth")
        self.latest_checkpoint_posttrain = os.path.join(self.checkpoints_dir, self.model_name + "_posttrain_latest.pth")
        self.best_checkpoint_posttrain = os.path.join(self.checkpoints_dir, self.model_name + "_posttrain_best.pth")

    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=1): 
        criterion = torch.nn.MSELoss()
        optimizer = optim.AdamW(self.nn.parameters(), lr=1e-3)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=epochs*len(train_loader)*0.1, total_steps=epochs*len(train_loader))
        scaler = GradScaler()
        best_val_loss = float('inf')
        start_epoch = 0

        logger = {
            "train_loss": [],
            "val_loss": [],
            "val_steps_interval": len(train_loader),
        }

        # load latest checkpoint if exists
        if os.path.exists(self.latest_checkpoint_pretrain):
            checkpoint = torch.load(self.latest_checkpoint_pretrain)
            self.nn.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            best_val_loss = checkpoint['best_val_loss']
            logger = checkpoint['logger']
            start_epoch = checkpoint['epoch']
            del checkpoint
            print("Loaded latest checkpoint")
        else:
            print("No latest checkpoint found, starting from scratch")

        for epoch in range(start_epoch, epochs):
            # ------------ Train ------------
            self.nn.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            total_loss = 0
            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)
                x = torch.fft.fftshift(torch.fft.fft2(x))
                x = torch.cat([torch.real(x), torch.imag(x)], dim=1)
                mean = x.mean()
                std = x.std() + 1e-6
                x = (x - mean) / std

                # Forward pass
                with autocast(self.device.type):
                    output_nn = self.nn(x)
                    s = self.size[-1] # H, W size
                    c = s // 2
                    r = self.radius
                    output_nn = torch.view_as_complex(output_nn.permute(0, 2, 3, 1).contiguous())
                    y_pred = torch.view_as_complex(torch.zeros_like(x).permute(0, 2, 3, 1).contiguous())
                    y_pred[:, c-r:c+r, c-r:c+r] = output_nn
                    y_pred = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_pred))).unsqueeze(1)
                    loss = criterion(y_pred, y)
                
                # Backward and optimize
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                logger["train_loss"].append(loss.item())
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}] average training loss: {avg_loss:.4f}")

            # ------------- Evaluation -----------------
            self.nn.eval()
            total_loss = 0
            total = 0
            with torch.no_grad(), autocast(self.device.type):
                progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
                for x, y in progress_bar:
                    x, y = x.to(self.device), y.to(self.device)
                    x = torch.fft.fftshift(torch.fft.fft2(x))
                    x = torch.cat([torch.real(x), torch.imag(x)], dim=1)
                    mean = x.mean()
                    std = x.std() + 1e-6
                    x = (x - mean) / std

                    with autocast(self.device.type):
                        output_nn = self.nn(x)
                        s = self.size[-1] # H, W size
                        c = s // 2
                        r = self.radius
                        output_nn = torch.view_as_complex(output_nn.permute(0, 2, 3, 1).contiguous())
                        y_pred = torch.view_as_complex(torch.zeros_like(x).permute(0, 2, 3, 1).contiguous())
                        y_pred[:, c-r:c+r, c-r:c+r] = output_nn
                        y_pred = torch.abs(torch.fft.ifft2(torch.fft.ifftshift(y_pred))).unsqueeze(1)
                        loss = criterion(y_pred, y)
                    
                    total_loss += loss.item()
                    
            avg_loss = total_loss / len(val_loader)
            logger["val_loss"].append(avg_loss)
            # print(f"Validation Loss: {avg_loss:.4f}")

            # ------------- Save checkpoint -----------------
            def save_checkpoint(path):
                torch.save(
                    {
                        "model": self.nn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch+1, # +1 because we save this value in the last iteration
                        "best_val_loss": best_val_loss,
                        "scaler": scaler.state_dict(),
                        "logger": logger
                    },
                    path)
            save_checkpoint(self.latest_checkpoint_pretrain)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                print(f"🟢 New best validation loss: {color.GREEN}{avg_loss:.4f}{color.RESET}")
                save_checkpoint(self.best_checkpoint_pretrain)
            else:
                print(f"🔴 Validation loss did not improve: {color.RED}{avg_loss:.4f}{color.RESET}")

    def train(self, train_loader, val_loader, epochs=1):
        print("🔴 Posttrain: Do nothing.")
        return 
        optimizer = optim.AdamW(self.nn.parameters(), lr=1e-4)
        scheduler = cosine_warmup_scheduler(optimizer, warmup_steps=epochs*len(train_loader)*0.1, total_steps=epochs*len(train_loader))
        scaler = GradScaler()
        best_val_loss = float('inf')
        start_epoch = 0

        logger = {
            "train_loss": [],
            "val_loss": [],
            "val_steps_interval": len(train_loader),
        }

        # load latest checkpoint if exists
        if os.path.exists(self.latest_checkpoint_posttrain):
            checkpoint = torch.load(self.latest_checkpoint_posttrain)
            self.nn.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            scaler.load_state_dict(checkpoint['scaler'])
            best_val_loss = checkpoint['best_val_loss']
            logger = checkpoint['logger']
            start_epoch = checkpoint['epoch']
            del checkpoint
            print("Loaded latest posttrain checkpoint")
        else:
            print("No latest checkpoint found, starting from pretrain model")
            self.nn.load_state_dict(torch.load(self.best_checkpoint_pretrain)['model'])

        for epoch in range(start_epoch, epochs):
            # ------------ Train ------------
            self.nn.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            total_loss = 0
            for target, _ in progress_bar:
                target = target.to(self.device)
                
                # Forward pass
                with autocast(self.device.type):
                    printedNom, printedMax, printedMin = self.simLitho(F.sigmoid(self.nn(target)).squeeze(1))
                    loss_l2 = F.mse_loss(printedNom.unsqueeze(1), target)
                    loss_pvb = F.mse_loss(printedMax, printedMin)
                    loss = loss_l2 + loss_pvb
                
                # Backward and optimize
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                logger["train_loss"].append(loss.item())
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}] average training loss: {avg_loss:.4f}")

            # ------------- Evaluation -----------------
            self.nn.eval()
            total_loss = 0
            total = 0
            with torch.no_grad(), autocast(self.device.type):
                progress_bar = tqdm(val_loader, desc="Evaluating", leave=False)
                for target, _ in progress_bar:
                    target = target.to(self.device)
                    printedNom, printedMax, printedMin = self.simLitho(F.sigmoid(self.nn(target)).squeeze(1))
                    loss_l2 = F.mse_loss(printedNom.unsqueeze(1), target)
                    loss_pvb = F.mse_loss(printedMax, printedMin)
                    loss = loss_l2 + loss_pvb
                    total_loss += loss.item()
                    total += target.size(0)
                    
            avg_loss = total_loss / len(val_loader)
            logger["val_loss"].append(avg_loss)
            # print(f"Validation Loss: {avg_loss:.4f}")

            # ------------- Save checkpoint -----------------
            def save_checkpoint(path):
                torch.save(
                    {
                        "model": self.nn.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch+1, # +1 because we save this value in the last iteration
                        "best_val_loss": best_val_loss,
                        "scaler": scaler.state_dict(),
                        "logger": logger
                    },
                    path)
            save_checkpoint(self.latest_checkpoint_posttrain)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                print(f"🟢 New best validation loss: {color.GREEN}{avg_loss:.4f}{color.RESET}")
                save_checkpoint(self.best_checkpoint_posttrain)
            else:
                print(f"🔴 Validation loss did not improve: {color.RED}{avg_loss:.4f}{color.RESET}")

    def run(self, target):
        self.nn.eval()
        with torch.no_grad(), autocast(self.device.type):
            target_fft = torch.fft.fftshift(torch.fft.fft(target))
            mask_fft_lowpass = self.nn(target_fft)[0, 0]
            mask_fft = torch.zeros((1, 1, 2048, 2048), device=self.device)
            center = 1024
            radius = 32
            mask_fft[:, :, center - radius:center + radius, center - radius:center + radius] = mask_fft_lowpass
            mask = torch.fft.ifft(torch.fft.ifftshift(mask_fft))
            return mask[0, 0].detach()

    def save(self, filenames):
        if os.path.exists(self.best_checkpoint_posttrain):
            best_nn = torch.load(self.best_checkpoint_posttrain)["model"]
        else:
            best_nn = torch.load(self.latest_checkpoint_pretrain)["model"]
        torch.save(best_nn, filenames)
        print(f"🟢 Saved best model to {filenames}")

    def load(self, filenames): 
        self.nn.load_state_dict(torch.load(filenames))
        print(f"🟢 Loaded model from {filenames}")

if __name__ == "__main__":
    """
    directly run this file to pretrain the model
    """
    train_loader, val_loader = loadersILT("MetalSet", (2048, 2048), batch_size=24, njobs=8)
    model = Freq01()
    model.pretrain(train_loader, val_loader, epochs=5)
    Folder = os.path.join("dev", "MetalSet_Freq01")
    model.evaluate("MetalSet", finetune=False, folder=Folder)
    