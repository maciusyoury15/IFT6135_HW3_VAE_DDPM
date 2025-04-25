
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())



class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=64):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, labels=None):
        x = self.maxpool_conv(x)
        if labels is not None:
            emb = self.emb_layer(labels)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x += emb
        #emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x #+ emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=64):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, labels):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        if labels is not None:
            emb = self.emb_layer(labels)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
            x += emb
        #emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x #+ emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, c_emb_dim=64, remove_deep_conv=False):
        super().__init__()
        self.c_emb_dim = c_emb_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 16)
        self.down1 = Down(16, 32)
        #self.sa1 = SelfAttention(32)
        self.down2 = Down(32, 64)
        #self.sa2 = SelfAttention(64)
        self.down3 = Down(64, 64)
        #self.sa3 = SelfAttention(64)


        if remove_deep_conv:
            self.bot1 = DoubleConv(64, 64)
            self.bot3 = DoubleConv(64, 64)
        else:
            self.bot1 = DoubleConv(64, 128)
            self.bot2 = DoubleConv(128, 128)
            self.bot3 = DoubleConv(128, 64)

        self.up1 = Up(128, 32)
        #self.sa4 = SelfAttention(32)
        self.up2 = Up(64, 16)
        #self.sa5 = SelfAttention(16)
        self.up3 = Up(32, 16)
        #self.sa6 = SelfAttention(16)
        self.outc = nn.Conv2d(16, c_out, kernel_size=1)


    def unet_forwad(self, x, labels):
        x1 = self.inc(x)
        x2 = self.down1(x1, labels)
        #x2 = self.sa1(x2)
        x3 = self.down2(x2, labels)
        #x3 = self.sa2(x3)
        x4 = self.down3(x3, labels)
        #x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, labels)
        #x = self.sa4(x)
        x = self.up2(x, x2, labels)
        #x = self.sa5(x)
        x = self.up3(x, x1, labels)
        #x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x):
        return self.unet_forwad(x)


class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, c_emb_dim=64, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, c_emb_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, c_emb_dim)

    def forward(self, x, y=None):
        if y is not None:
            y = self.label_emb(y)

        return self.unet_forwad(x, y)