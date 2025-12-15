"""
unet.py + OptiCo
"""
import sys
sys.path.append(".")
from lithobench.dataset import *
from lithobench.model import *
import pylitho.exact as litho
import torch.optim as optim
import math
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast, GradScaler
from mine.utils import color

class LayerNorm2d(nn.Module):
    """对 NCHW 做 LayerNorm（按通道归一化，等价于 channels-last LayerNorm）。"""
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # affine
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


def complex_mul(ar, ai, br, bi):
    """(a_r + j a_i) * (b_r + j b_i)"""
    return ar * br - ai * bi, ar * bi + ai * br


class OptiCoBlockEq567(nn.Module):
    """
    按论文 Eq.(5)(6)(7) 实现的 OptiCo Block（NCHW）。

    Eq.(5): Y_backbone(U) = ( DWConv( Norm(Ur)W1 ) ⊙ σ( Norm(Ur)W2 ) ) W3 + Ur
    Eq.(6): Y_phase(U) = [ ComplexConv1D( Y_backbone(U) ) ⊙ OPconv(U) ] * exp(jkz)/(j*λ*z)
    Eq.(7): Y_OptiCo = Y_backbone(U) + Y_phase(U)

    注意：输出需要与 backbone 的实数特征对齐，所以这里将 Y_phase 取实部后加回。
    """
    def __init__(
        self,
        channels: int,
        op_kernel_size: int = 11,
        z: float = 1.0,
        wavelength: float = 1.0,
        dw_kernel_size: int = 7,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert op_kernel_size % 2 == 1, "OP kernel size 建议用奇数以便 same padding"

        self.c = channels
        self.ks = op_kernel_size
        self.z = float(z)
        self.wavelength = float(wavelength)
        self.eps = eps

        # 物理波数 k = 2π/λ
        k = 2.0 * math.pi / max(self.wavelength, eps)
        self.register_buffer("k", torch.tensor(k, dtype=torch.float32))

        # --- Eq.(5) backbone 分支参数 ---
        self.norm = LayerNorm2d(channels, eps=eps)

        # DWConv
        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=dw_kernel_size,
            padding=dw_kernel_size // 2,
            groups=channels,
            bias=True,
        )
        # W1, W2, W3 都是 pointwise（1x1）映射
        self.w1 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.w2 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.w3 = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

        # --- OP kernel + learnable scalar alpha（缩放 OP kernel）---
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))  # learnable scalar weight in paper
        q_real, q_imag = self._build_qpf_kernel(self.ks, k, self.z)         # Q(x,y)
        self.register_buffer("q_real", q_real)  # (1,1,ks,ks)
        self.register_buffer("q_imag", q_imag)  # (1,1,ks,ks)

        # --- ComplexConv1D：逐像素复数 embedding（用 1x1 产生 real/imag）---
        # ComplexConv1D(Y_backbone): 输出 complex feature (Fr, Fi)
        self.complex_embed = nn.Conv2d(channels, 2 * channels, kernel_size=1, bias=True)

        # 相位分支整体强度（可学，工程上很常见；不改变 Eq.(6) 结构，只是缩放）
        self.phase_gain = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    @staticmethod
    def _build_qpf_kernel(kernel_size: int, k: float, z: float):
        """
        Fresnel QPF: exp( j*k/(2z) * (x^2 + y^2) )
        返回 (q_real, q_imag) 形状均 (1,1,ks,ks)
        """
        N = kernel_size
        center = (N - 1) / 2.0
        xs = torch.arange(N, dtype=torch.float32) - center
        ys = torch.arange(N, dtype=torch.float32) - center
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        phase = (k / (2.0 * max(z, 1e-6))) * (X**2 + Y**2)
        q_real = torch.cos(phase)[None, None, :, :]
        q_imag = torch.sin(phase)[None, None, :, :]
        return q_real, q_imag

    def _opconv(self, ur: torch.Tensor, ui: torch.Tensor | None = None):
        """
        Eq.(4) 的 OP complex convolution:
        OPconv(U) = (Ur*Wr - Ui*Wi) + j(Ur*Wi + Ui*Wr)

        这里用 depthwise conv：每个通道用同一个 OP kernel。
        """
        if ui is None:
            ui = torch.zeros_like(ur)

        B, C, H, W = ur.shape
        assert C == self.c

        q_real = self.q_real.to(device=ur.device, dtype=ur.dtype)
        q_imag = self.q_imag.to(device=ur.device, dtype=ur.dtype)

        # Weff = alpha * Q
        wr = (self.alpha.to(dtype=ur.dtype) * q_real).repeat(C, 1, 1, 1)  # (C,1,ks,ks)
        wi = (self.alpha.to(dtype=ur.dtype) * q_imag).repeat(C, 1, 1, 1)

        pad = self.ks // 2
        real = F.conv2d(ur, wr, padding=pad, groups=C) - F.conv2d(ui, wi, padding=pad, groups=C)
        imag = F.conv2d(ur, wi, padding=pad, groups=C) + F.conv2d(ui, wr, padding=pad, groups=C)
        return real, imag

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B,C,H,W) 实数输入（视为 Ur），输出同形状实数特征。
        """
        ur = u
        # --- Eq.(5): Y_backbone(U) ---
        n = self.norm(ur)
        a = self.dw(n)
        a = self.w1(a)

        g = self.sigmoid(self.w2(n))
        y_backbone = self.w3(a * g) + ur  # residual + Ur

        # --- Eq.(6): Y_phase(U) ---
        # ComplexConv1D( Y_backbone(U) ) -> (Fr, Fi)
        emb = self.complex_embed(y_backbone)
        fr, fi = torch.chunk(emb, 2, dim=1)

        # OPconv(U) -> (Pr, Pi)
        pr, pi = self._opconv(ur, ui=None)

        # Hadamard product in complex domain
        mr, mi = complex_mul(fr, fi, pr, pi)

        # multiply by exp(jkz)/(j*λ*z)  (λ here is wavelength)
        k = self.k.to(device=u.device, dtype=u.dtype)
        z = torch.tensor(self.z, device=u.device, dtype=u.dtype)
        lam = torch.tensor(self.wavelength, device=u.device, dtype=u.dtype)

        # exp(jkz)
        cos_kz = torch.cos(k * z)
        sin_kz = torch.sin(k * z)

        # divide by (j*λ*z): 1/(jA) = -j/A
        A = torch.clamp(lam * z, min=torch.tensor(self.eps, device=u.device, dtype=u.dtype))
        # (mr + j mi) * (cos + j sin)
        tr, ti = complex_mul(mr, mi, cos_kz, sin_kz)
        # multiply by (-j/A): (tr + j ti) * (0 - j/A) = (ti/A) + j(-tr/A)
        ypr = (ti / A)
        ypi = (-tr / A)

        # Eq.(7): Y_OptiCo = Y_backbone + Y_phase
        # 由于 Y_backbone 是实数，这里将 Y_phase 投影回实数（取实部）后相加
        y_optico = y_backbone + self.phase_gain * ypr
        return y_optico


class DoubleConvOptiCo(nn.Module):
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
        self.OptiCo = OptiCoBlockEq567(out_channels)

    def forward(self, x):
        down = self.conv(x)
        out = self.OptiCo(down)
        return out
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x

class OptiCo_Unetnn(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConvOptiCo(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConvOptiCo(features[-1], features[-1]*2)

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                UpConv(
                    feature*2, feature
                )
            )
            self.ups.append(DoubleConvOptiCo(feature*2, feature))

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
    
class OptiCo_Unet(ModelILT):
    def __init__(self, size=(256, 256)): 
        super().__init__(size=size, name=self.__class__.__name__)
        self.simLitho = litho.LithoSim("./config/lithosimple.txt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = OptiCo_Unetnn(in_channels=1, out_channels=1, features=[64, 128, 256, 512]).to(self.device)

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
    MODEL_NAME = "OptiCo_Unet"

    train_loader, val_loader = loadersILT(TRAIN_DATASET, IMAGE_SIZE, batch_size=BATCH_SIZE, njobs=8)
    model = OptiCo_Unet(size=IMAGE_SIZE)
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

