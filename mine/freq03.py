import sys
sys.path.append(".")
from lithobench.model import *
from lithobench.dataset import *
import pylitho.exact as litho
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from mine.utils import color

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class Freq03nn(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                UpConv(
                    feature*2, feature
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True
                )

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
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
    
class Freq03(ModelILT):
    def __init__(self, size=(256, 256)): 
        super().__init__(size=size, name=self.__class__.__name__)
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = Freq03nn(in_channels=2, out_channels=1, features=[64, 128, 256, 512]).to(self.device)

        self.model_name = self.__class__.__name__
        self.checkpoints_dir = "./mine/checkpoints"
        self.latest_checkpoint_pretrain = os.path.join(self.checkpoints_dir, self.model_name + "_latest.pth")
        self.best_checkpoint_pretrain = os.path.join(self.checkpoints_dir, self.model_name + "_best.pth")
        self.latest_checkpoint_posttrain = os.path.join(self.checkpoints_dir, self.model_name + "_latest_posttrain.pth")
        self.best_checkpoint_posttrain = os.path.join(self.checkpoints_dir, self.model_name + "_best_posttrain.pth")

    @property
    def size(self): 
        return self._size
    @property
    def name(self): 
        return self._name

    def pretrain(self, train_loader, val_loader, epochs=40): 
        criterion = F.binary_cross_entropy_with_logits
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
            for target, mask in progress_bar:
                target, mask = target.to(self.device), mask.to(self.device)
                target_t = torch.fft.fft2(target)
                target_t = torch.fft.fftshift(target_t)
                target_t = torch.real(torch.fft.ifft(target_t))
                target = torch.cat((target, target_t), dim=1)
                
                # Forward pass
                with autocast(self.device.type):
                    outputs = self.nn(target)
                    loss = criterion(outputs, mask)
                
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
                for target, mask in progress_bar:
                    target, mask = target.to(self.device), mask.to(self.device)
                    target_t = torch.fft.fft2(target)
                    target_t = torch.fft.fftshift(target_t)
                    target_t = torch.real(torch.fft.ifft(target_t))
                    target = torch.cat((target, target_t), dim=1)
                    outputs = self.nn(target)
                    loss = criterion(outputs, mask)
                    total_loss += loss.item()
                    total += mask.size(0)
                    
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

    def train(self, train_loader, val_loader, epochs=40):
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
                target_t = torch.fft.fft2(target)
                target_t = torch.fft.fftshift(target_t)
                target_t = torch.real(torch.fft.ifft(target_t))
                target_t = torch.cat((target, target_t), dim=1)
                
                # Forward pass
                with autocast(self.device.type):
                    printedNom, printedMax, printedMin = self.simLitho(F.sigmoid(self.nn(target_t)).squeeze(1))
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
                    target_t = torch.fft.fft2(target)
                    target_t = torch.fft.fftshift(target_t)
                    target_t = torch.real(torch.fft.ifft(target_t))
                    target_t = torch.cat((target, target_t), dim=1)
                    printedNom, printedMax, printedMin = self.simLitho(F.sigmoid(self.nn(target_t)).squeeze(1))
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
            target_t = torch.fft.fft2(target)
            target_t = torch.fft.fftshift(target_t)
            target_t = torch.real(torch.fft.ifft(target_t))
            target = torch.cat((target, target_t), dim=1)
            return torch.sigmoid(self.nn(target)[0, 0]).detach()

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
    directly run this file to train the model
    """
    BATCH_SIZE = 24
    EPOCHS = 1
    IMAGE_SIZE = (256, 256)
    TRAIN_DATASET = "MetalSet"
    TEST_DATASET = "StdMetal"
    MODEL_NAME = "Freq03"

    train_loader, val_loader = loadersILT(TRAIN_DATASET, IMAGE_SIZE, batch_size=BATCH_SIZE, njobs=8)
    model = Freq03(size=IMAGE_SIZE)
    Folder = os.path.join("dev", f"{TRAIN_DATASET}_{MODEL_NAME}")

    # evaluate the randomly initialized model
    targets = evaluate.getTargets(samples=None, dataset=TRAIN_DATASET)
    model.evaluate(targets, finetune=False, folder=Folder)

    # evaluate the pretrained model
    model.pretrain(train_loader, val_loader, epochs=EPOCHS)
    model.evaluate(targets, finetune=False, folder=Folder)

    # evaluate the posttrained model
    model.train(train_loader, val_loader, epochs=EPOCHS)
    model.evaluate(targets, finetune=False, folder=Folder)

    # evaluate on StdMetal
    Folder = os.path.join("saved", f"{TEST_DATASET}_{MODEL_NAME}")
    targets = evaluate.getTargets(samples=None, dataset=TEST_DATASET)
    model.evaluate(targets, finetune=False, folder=Folder)