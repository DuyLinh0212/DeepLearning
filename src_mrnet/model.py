import torch
import torch.nn as nn
from torchvision import models


def replace_bn_with_gn(model, num_groups=8):
    """
    Đệ quy thay tất cả BatchNorm2d bằng GroupNorm.
    Chọn số groups lớn nhất chia hết num_features, tối thiểu là 2 để tránh
    suy biến thành InstanceNorm (groups=1) gây mất ổn định khi batch_size=1.
    Nếu num_features < 2 thì giữ nguyên BatchNorm.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            num_features = module.num_features
            # Tìm số groups lớn nhất chia hết num_features, nhưng tối thiểu là 2
            groups = num_groups
            while num_features % groups != 0:
                groups -= 1
            if groups < 2:
                # Không thể tạo GroupNorm hợp lệ, giữ nguyên BatchNorm
                continue
            gn = nn.GroupNorm(num_groups=groups, num_channels=num_features,
                              affine=True)
            # Copy lại weight/bias đã học từ pretrained để không mất thông tin
            gn.weight.data.copy_(module.weight.data)
            gn.bias.data.copy_(module.bias.data)
            setattr(model, name, gn)
        else:
            replace_bn_with_gn(module, num_groups)
    return model


def _build_backbone(backbone, training):
    """Tạo một backbone đơn lẻ và trả về (net, feature_dim)."""
    if backbone == "resnet18":
        weights = None
        if training and hasattr(models, "ResNet18_Weights"):
            weights = models.ResNet18_Weights.DEFAULT
        net = models.resnet18(weights=weights) if weights is not None \
            else models.resnet18(pretrained=training)
        # Bỏ lớp FC cuối, giữ AdaptiveAvgPool → output shape (B, 512, 1, 1)
        net = nn.Sequential(*list(net.children())[:-1])
        for param in net.parameters():
            param.requires_grad = False
        feature_dim = 512

    elif backbone == "alexnet":
        if hasattr(models, "AlexNet_Weights"):
            weights = models.AlexNet_Weights.DEFAULT if training else None
            net = models.alexnet(weights=weights)
        else:
            net = models.alexnet(pretrained=training)
        feature_dim = 256  # output của features[-1] sau GAP

    elif backbone == "efficientnet_b0":
        if hasattr(models, "EfficientNet_B0_Weights"):
            weights = models.EfficientNet_B0_Weights.DEFAULT if training else None
            net = models.efficientnet_b0(weights=weights)
        else:
            net = models.efficientnet_b0(pretrained=training)
        # Thay BN → GN để hoạt động đúng với batch_size=1
        net = replace_bn_with_gn(net, num_groups=8)
        feature_dim = 1280  # output của features[-1] sau GAP

    else:
        raise ValueError(f"Backbone không hỗ trợ: {backbone}")

    return net, feature_dim


class MRNet(nn.Module):
    """Model đơn giản dùng AlexNet cho một mặt phẳng."""
    def __init__(self):
        super().__init__()
        if hasattr(models, "AlexNet_Weights"):
            self.model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        else:
            self.model = models.alexnet(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)  # batch_size=1 only
        x = self.model.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x


class TripleMRNet(nn.Module):
    """
    Ba backbone song song (axial / sagittal / coronal), đặc trưng được
    max-pooled theo chiều slice rồi nối lại cho classifier.
    """
    def __init__(self, backbone="efficientnet_b0", training=True):
        super().__init__()
        self.backbone = backbone

        self.axial_net, feature_dim = _build_backbone(backbone, training)
        self.sagit_net, _           = _build_backbone(backbone, training)
        self.coron_net, _           = _build_backbone(backbone, training)

        self.gap_axial = nn.AdaptiveAvgPool2d(1)
        self.gap_sagit = nn.AdaptiveAvgPool2d(1)
        self.gap_coron = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(3 * feature_dim, 1)

    def _extract(self, net, vol):
        """Trích xuất feature từ một volume (nhiều slice)."""
        if self.backbone == "resnet18":
            # resnet18 đã bị bỏ FC, output là (B,512,1,1)
            feat = net(vol).view(vol.size(0), -1)
        else:
            # alexnet / efficientnet: dùng .features rồi GAP
            feat = net.features(vol)
            if self.backbone == "alexnet":
                feat = self.gap_axial(feat).view(feat.size(0), -1) \
                    if net is self.axial_net else \
                    (self.gap_sagit(feat).view(feat.size(0), -1)
                     if net is self.sagit_net else
                     self.gap_coron(feat).view(feat.size(0), -1))
            else:
                pool = (self.gap_axial if net is self.axial_net else
                        self.gap_sagit if net is self.sagit_net else
                        self.gap_coron)
                feat = pool(feat).view(feat.size(0), -1)
        return feat

    def forward(self, vol_axial, vol_sagit, vol_coron):
        vol_axial = torch.squeeze(vol_axial, dim=0)
        vol_sagit = torch.squeeze(vol_sagit, dim=0)
        vol_coron = torch.squeeze(vol_coron, dim=0)

        if self.backbone == "resnet18":
            fa = self.axial_net(vol_axial).view(vol_axial.size(0), -1)
            fs = self.sagit_net(vol_sagit).view(vol_sagit.size(0), -1)
            fc = self.coron_net(vol_coron).view(vol_coron.size(0), -1)
        else:
            fa = self.gap_axial(self.axial_net.features(vol_axial)).view(vol_axial.size(0), -1)
            fs = self.gap_sagit(self.sagit_net.features(vol_sagit)).view(vol_sagit.size(0), -1)
            fc = self.gap_coron(self.coron_net.features(vol_coron)).view(vol_coron.size(0), -1)

        x = torch.max(fa, 0, keepdim=True)[0]
        y = torch.max(fs, 0, keepdim=True)[0]
        z = torch.max(fc, 0, keepdim=True)[0]

        out = self.classifier(torch.cat((x, y, z), dim=1))
        return out