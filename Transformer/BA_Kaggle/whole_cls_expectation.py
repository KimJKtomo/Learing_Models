
import argparse
import numpy as np
import torch
import timm
from PIL import Image
import torchvision.transforms as T


def build_uniform_bin_centers(class_n: int, y_max_months: float = 240.0):
    # default: uniform centers (0..y_max) with class_n bins
    edges = np.linspace(0.0, y_max_months, class_n + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers


class WholeClassifier(torch.nn.Module):
    def __init__(self, model_name: str, class_n: int, pretrained: bool = False):
        super().__init__()
        # NOTE: 여기 model_name은 BA_classification에서 사용한 args.model 값이 아니라,
        # timm 모델 id로 직접 넘기는 버전.
        self.model = timm.create_model(model_name, pretrained=pretrained)
        # timm models: reset_classifier(num_classes)
        self.model.reset_classifier(class_n)

    def forward(self, x):
        return self.model(x)


def load_state_dict_flexible(model, ckpt_path: str, device: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    # some ckpts may store {'state_dict': ...}
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


def predict_month_and_conf(model, img_tensor, bin_centers, device):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        prob = torch.softmax(logits, dim=1)
        conf = prob.max(dim=1).values
        month = (prob * torch.tensor(bin_centers, device=device).view(1, -1)).sum(dim=1)
    return float(month.item()), float(conf.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=str)
    ap.add_argument("--timm_name", required=True, type=str,
                    help="timm model id (e.g., swin_large_patch4_window12_384_in22k)")
    ap.add_argument("--class_n", default=34, type=int)
    ap.add_argument("--img", required=True, type=str)
    ap.add_argument("--img_size", default=384, type=int)
    ap.add_argument("--y_max", default=240.0, type=float)
    ap.add_argument("--device", default="cuda", type=str)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model = WholeClassifier(args.timm_name, args.class_n, pretrained=False).to(device)
    missing, unexpected = load_state_dict_flexible(model, args.ckpt, device)
    print("[LOAD]", args.ckpt)
    print("  missing:", len(missing), "unexpected:", len(unexpected))

    tfm = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
        # normalize to imagenet stats (if your training used different norm, match it)
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(args.img).convert("RGB")
    x = tfm(img).unsqueeze(0)

    centers = build_uniform_bin_centers(args.class_n, args.y_max)
    month, conf = predict_month_and_conf(model, x, centers, device)
    print("pred_month:", month, "conf:", conf)


if __name__ == "__main__":
    main()
