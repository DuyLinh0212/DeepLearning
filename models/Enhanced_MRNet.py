import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

class SEBlock(nn.Module):
    """Khoi SE (Squeeze-and-Excitation) cho co che chu y theo kenh."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class MRnet(nn.Module):
    """MRnet dung EfficientNet-B0 pretrained lam backbone de trich xuat dac trung."""

    def __init__(self):
        """Khoi tao mo hinh MRnet."""
        super(MRnet, self).__init__()

        # Tao 3 backbone cho 3 mat phang
        self.axial = self._make_backbone()
        self.coronal = self._make_backbone()
        self.sagittal = self._make_backbone()

        # SE cho tung mat phang
        self.se_axial = SEBlock(channels=1280)
        self.se_coronal = SEBlock(channels=1280)
        self.se_sagittal = SEBlock(channels=1280)

        # Adaptive Avg Pooling de gom dac trung theo kenh
        self.pool_axial = nn.AdaptiveAvgPool2d(1)
        self.pool_coronal = nn.AdaptiveAvgPool2d(1)
        self.pool_sagittal = nn.AdaptiveAvgPool2d(1)

        # FC cuoi de du doan xac suat benh
        self.fc = nn.Sequential(
            nn.Linear(in_features=3 * 1280, out_features=1)
        )

    def _make_backbone(self):
        weights = EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        return backbone.features

    def forward(self, axial, coronal=None, sagittal=None):
        """Dau vao co the la bo 3 tensor (axial, coronal, sagittal) hoac list/tuple 3 phan tu.

        Moi tensor co dang [1, slices, 3, 224, 224] khi batch_size=1.
        """

        # Ho tro dau vao dang list/tuple
        if coronal is None and sagittal is None:
            if isinstance(axial, (list, tuple)) and len(axial) == 3:
                axial, coronal, sagittal = axial
            else:
                raise ValueError("Dau vao phai la (axial, coronal, sagittal) hoac list/tuple 3 phan tu.")

        # Bo dim batch khi batch_size=1
        images = [torch.squeeze(img, dim=0) for img in (axial, coronal, sagittal)]

        # Trich xuat dac trung tu 3 mat phang
        image1 = self.axial(images[0])
        image2 = self.coronal(images[1])
        image3 = self.sagittal(images[2])

        # Ap dung SE
        image1 = self.se_axial(image1)
        image2 = self.se_coronal(image2)
        image3 = self.se_sagittal(image3)

        # Chuyen [slices, 1280, 1, 1] -> [slices, 1280]
        image1 = self.pool_axial(image1).view(image1.size(0), -1)
        image2 = self.pool_coronal(image2).view(image2.size(0), -1)
        image3 = self.pool_sagittal(image3).view(image3.size(0), -1)

        # Max pooling theo slice de lay dac trung noi bat
        image1 = torch.max(image1, dim=0, keepdim=True)[0]
        image2 = torch.max(image2, dim=0, keepdim=True)[0]
        image3 = torch.max(image3, dim=0, keepdim=True)[0]

        # Ghep 3 mat phang thanh vector [1, 1280*3]
        output = torch.cat([image1, image2, image3], dim=1)

        # Dua vao FC de ra xac suat
        output = self.fc(output)
        return output

    def _load_wieghts(self):
        """Nap trong so pretrained (neu can)."""
        pass
