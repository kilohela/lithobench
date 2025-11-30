from lithobench.model import *
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

class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels, out_channels, deep_supervision=False):
        super().__init__()
        nb_filter = [32, 64, 128, 256, 512] # number of filters in each layer
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Encoder
        self.conv0_0 = DoubleConv(in_channels, nb_filter[0])
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1])
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2])
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3])
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4])

        # Decoder (nested)
        self.conv0_1 = DoubleConv(nb_filter[0]+nb_filter[1], nb_filter[0])
        self.conv1_1 = DoubleConv(nb_filter[1]+nb_filter[2], nb_filter[1])
        self.conv2_1 = DoubleConv(nb_filter[2]+nb_filter[3], nb_filter[2])
        self.conv3_1 = DoubleConv(nb_filter[3]+nb_filter[4], nb_filter[3])

        self.conv0_2 = DoubleConv(nb_filter[0]*2+nb_filter[1], nb_filter[0])
        self.conv1_2 = DoubleConv(nb_filter[1]*2+nb_filter[2], nb_filter[1])
        self.conv2_2 = DoubleConv(nb_filter[2]*2+nb_filter[3], nb_filter[2])

        self.conv0_3 = DoubleConv(nb_filter[0]*3+nb_filter[1], nb_filter[0])
        self.conv1_3 = DoubleConv(nb_filter[1]*3+nb_filter[2], nb_filter[1])

        self.conv0_4 = DoubleConv(nb_filter[0]*4+nb_filter[1], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final(x0_4)
            return output
    
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
    
class UPP(ModelILT):
    def __init__(self, size=(256, 256)): 
        super().__init__(size=size, name="DAMOILT")
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = UNetPlusPlus(in_channels=1, out_channels=1).to(self.device)

        self.model_name = "unet_pp"
        self.checkpoints_dir = "./mine/checkpoints"
        self.latest_checkpoint = os.path.join(self.checkpoints_dir, self.model_name + "_latest.pth")
        self.best_checkpoint = os.path.join(self.checkpoints_dir, self.model_name + "_best.pth")

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
        if os.path.exists(self.latest_checkpoint):
            checkpoint = torch.load(self.latest_checkpoint)
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
            save_checkpoint(self.latest_checkpoint)
            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                print(f"🟢 New best validation loss: {color.GREEN}{avg_loss:.4f}{color.RESET}")
                save_checkpoint(self.best_checkpoint)
            else:
                print(f"🔴 Validation loss did not improve: {color.RED}{avg_loss:.4f}{color.RESET}")

    def train(self, train_loader, val_loader, epochs=1):
        print(f"{color.YELLOW}[WARNING] Training is not implemented yet. Please implement the train method in {self.__class__.__name__}{color.RESET}")
        pass

    def run(self, target):
        self.nn.eval()
        with torch.no_grad(), autocast(self.device.type):
            return torch.sigmoid(self.nn(target)[0, 0]).detach()

    def save(self, filenames):
        best_nn = torch.load(self.best_checkpoint)["model"]
        torch.save(best_nn, filenames)
        print(f"🟢 Saved best model to {filenames}")

    def load(self, filenames): 
        self.nn.load_state_dict(torch.load(filenames))
        print(f"🟢 Loaded model from {filenames}")