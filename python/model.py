import torch.nn as nn
import torch
import torch.nn.functional as F
import time

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class unet(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(unet, self).__init__()
        self.n_channels = 3
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)#256,32,32
        x = self.up2(x, x3)#128,64,64
        x = self.up3(x, x2)#64,128,128
        x = self.up4(x, x1)#64,256,256
        logits = self.outc(x)
        out_softmax = F.softmax(logits, dim=1)
        
        return out_softmax
    
if __name__ == "__main__":
    model = unet(4)
    model_path = r"best_epoch_weights.pth"  # 您的 PyTorch 模型路径
    model.load_state_dict(torch.load(model_path))
    model.eval()
    totol_time = 0
    for i in range(10):
        input_shape = (1, 3, 256, 256)  # 定义输入形状
        input_data = torch.randn(input_shape)
        start_time = time.time()
        # 使用 GPU 运行模型（如果可用）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        input_data = input_data.to(device)
        with torch.no_grad():
            output = model(input_data)
        output = output.cpu()
        end_time = time.time()
        totol_time += (end_time - start_time)
        print("Output shape:", output.shape)
        print("Inference time:", end_time - start_time, "seconds")
    print("mainInference time:", (totol_time)/10, "seconds")