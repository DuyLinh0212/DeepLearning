import torch
from torch.utils.data import DataLoader
from dataset import MRDataset
from torchvision import models
from models.Enhanced_MRNet import MRnet


def main():
    # Neu khong muon tai weights, bat NoPretrain = True
    NoPretrain = True
    # Giam tai bo nho de test nhanh
    TargetSlices = 8
    ImageSize = 128
    # Chi can output mau thi de False (giam RAM)
    DoTrainStep = False

    print("Bat dau khoi tao mo hinh...")
    if NoPretrain:
        class MRnetNoPretrain(MRnet):
            def _make_backbone(self):
                return models.efficientnet_b0(weights=None).features
        model = MRnetNoPretrain()
    else:
        model = MRnet()
    print("Da khoi tao mo hinh.")

    print("Dang khoi tao dataset...")
    dataset = MRDataset(task="abnormal", split="valid", target_slices=TargetSlices, image_size=ImageSize, augment=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print("Da khoi tao dataset va dataloader.")

    print("Dang lay 1 batch...")
    x, y = next(iter(loader))
    print("Da lay batch.")
    print("Dang chay forward...")
    with torch.no_grad():
        logits = model(x)
    print("Da forward xong.")

    print("Output shape:", logits.shape)
    print("Output sample:", logits.detach().cpu().numpy())

    # Thu 1 buoc hoc
    if DoTrainStep:
        print("Dang tinh loss va backprop...")
        loss_fn = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)

        logits_train = model(x)
        loss = loss_fn(logits_train, y)
        loss.backward()
        opt.step()
        print("Da cap nhat 1 buoc.")

        print("Loss:", float(loss.item()))


if __name__ == "__main__":
    main()
