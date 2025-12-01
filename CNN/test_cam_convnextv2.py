import os
import argparse
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

import timm

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ============================================================
# Dataset
# ============================================================

IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(root: str) -> List[str]:
    paths = []
    for ext in IMG_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
    return sorted(paths)


class SimpleImageDataset(Dataset):
    def __init__(self, img_paths: List[str], img_size: int = 224):
        self.img_paths = img_paths
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        self.resize_only = T.Resize((img_size, img_size))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        img_resized = self.resize_only(img)  # overlay용
        tensor = self.transform(img)
        return tensor, img_resized, p


# ============================================================
# CAM Hook
# ============================================================

class CAMHook:
    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        # output: [B, C, H, W]
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output[0]: [B, C, H, W]
        self.gradients = grad_output[0].detach()

    def remove(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


# ============================================================
# CAM 계산
# ============================================================

def compute_gradcam(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    acts: [B, C, H, W]
    grads: [B, C, H, W]
    return: [B, H, W] in [0,1]
    """
    B, C, H, W = acts.shape
    weights = grads.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    cam = (weights * acts).sum(dim=1)               # [B, H, W]
    cam = torch.relu(cam)

    cam_ = []
    for i in range(B):
        m = cam[i]
        m_min, m_max = m.min(), m.max()
        if (m_max - m_min) < 1e-8:
            cam_.append(torch.zeros_like(m))
        else:
            cam_.append((m - m_min) / (m_max - m_min))
    cam = torch.stack(cam_, dim=0)
    return cam


def compute_gradcampp(acts: torch.Tensor, grads: torch.Tensor) -> torch.Tensor:
    """
    Grad-CAM++
    acts: [B, C, H, W]
    grads: [B, C, H, W]
    """
    B, C, H, W = acts.shape

    grads2 = grads ** 2
    grads3 = grads ** 3

    sum_grads2 = grads2.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
    sum_grads3 = grads3.sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

    eps = 1e-8
    alpha_num = grads2
    alpha_den = 2 * grads2 + acts * sum_grads3
    alpha_den = torch.where(alpha_den != 0.0, alpha_den, torch.ones_like(alpha_den) * eps)
    alpha = alpha_num / alpha_den  # [B, C, H, W]

    positive_grad = torch.relu(grads)
    weights = (alpha * positive_grad).sum(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

    cam = (weights * acts).sum(dim=1)  # [B, H, W]
    cam = torch.relu(cam)

    cam_ = []
    for i in range(B):
        m = cam[i]
        m_min, m_max = m.min(), m.max()
        if (m_max - m_min) < 1e-8:
            cam_.append(torch.zeros_like(m))
        else:
            cam_.append((m - m_min) / (m_max - m_min))
    cam = torch.stack(cam_, dim=0)
    return cam


def compute_eigencam(acts: torch.Tensor) -> torch.Tensor:
    """
    Eigen-CAM (간단 구현)
    acts: [B, C, H, W]
    return: [B, H, W]
    """
    B, C, H, W = acts.shape
    cams = []
    for b in range(B):
        A = acts[b]  # [C, H, W]
        A_flat = A.view(C, -1)  # [C, H*W]
        A_mean = A_flat.mean(dim=1, keepdim=True)
        A_flat_centered = A_flat - A_mean

        cov = A_flat_centered @ A_flat_centered.t() / (A_flat_centered.shape[1] - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        v = eigvecs[:, -1]  # 가장 큰 고유값

        cam_flat = torch.matmul(v.unsqueeze(0), A_flat_centered).view(H, W)
        cam_flat = torch.relu(cam_flat)

        m_min, m_max = cam_flat.min(), cam_flat.max()
        if (m_max - m_min) < 1e-8:
            cam_norm = torch.zeros_like(cam_flat)
        else:
            cam_norm = (cam_flat - m_min) / (m_max - m_min)
        cams.append(cam_norm)
    return torch.stack(cams, dim=0)  # [B, H, W]


def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    img: PIL Image (RGB, 이미 리사이즈 된 상태)
    cam: [H, W] numpy, 0~1
    """
    img_np = np.array(img).astype(np.float32)

    cam_resized = cam
    if cam_resized.shape[0] != img_np.shape[0] or cam_resized.shape[1] != img_np.shape[1]:
        cam_resized = cv2.resize(cam_resized, (img_np.shape[1], img_np.shape[0]))

    cam_norm = (cam_resized * 255).astype(np.uint8)

    if _HAS_CV2:
        heatmap = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32)
        overlay = heatmap * alpha + img_np * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    else:
        heatmap = np.zeros_like(img_np)
        heatmap[..., 0] = cam_norm  # R 채널
        overlay = heatmap * alpha + img_np * (1 - alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)


# ============================================================
# ConvNeXt V2 모델 빌드
# ============================================================

def build_model(model_name: str,
                weight_path: str = None,
                device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, nn.Module]:
    """
    timm 기반 ConvNeXt V2 포함 모델 생성 + 가중치 로드
    마지막 Conv2d를 CAM 타깃 레이어로 사용
    """
    print(f"[INFO] Building model: {model_name}")

    # pytorch-image-models 설치되어 있으면 여기서 convnextv2_* 생성 가능
    model = timm.create_model(model_name, pretrained=(not bool(weight_path)))

    if weight_path:
        print(f"[INFO] Loading weight: {weight_path}")
        state = torch.load(weight_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[WARN] Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(device)

    # ConvNeXt V2 구조에서 마지막 Conv2d 찾기
    target_layer = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target_layer = m
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in the model for CAM.")

    print(f"[INFO] Using target layer for CAM: {target_layer}")
    return model, target_layer


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="MVTec CAM with ConvNeXt V2")

    p.add_argument("--gpu", type=int, default=0, help="GPU id to use. -1 for CPU.")
    p.add_argument("--model", type=str, default="convnextv2_tiny.fcmae_ft_in22k_in1k",
                   help="timm model name (e.g., convnextv2_tiny.fcmae_ft_in22k_in1k).")
    p.add_argument("--data_dir", type=str, required=True, help="Input image root directory (MVTec category root).")
    p.add_argument("--save_dir", type=str, required=True, help="Output directory to save CAM images.")
    p.add_argument("--img_size", type=int, default=224, help="Input image size.")
    p.add_argument("--cam_type", type=str, default="gradcam",
                   choices=["gradcam", "gradcampp", "eigencam"], help="CAM method.")
    p.add_argument("--weight_path", type=str, default="", help="Path to model weights (.pth/.pt). Empty => timm pretrained")
    p.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    p.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")

    return p.parse_args()


def main():
    args = parse_args()

    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    img_paths = list_images(args.data_dir)
    if len(img_paths) == 0:
        print(f"[WARN] No images found under: {args.data_dir}")
        return

    print(f"[INFO] Found {len(img_paths)} images.")

    dataset = SimpleImageDataset(img_paths, img_size=args.img_size)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    weight_path = args.weight_path if args.weight_path.strip() != "" else None
    model, target_layer = build_model(args.model,
                                      weight_path=weight_path,
                                      device=device)
    cam_hook = CAMHook(target_layer)

    os.makedirs(args.save_dir, exist_ok=True)

    model.eval()

    with torch.enable_grad():
        for batch_idx, (images, resized_imgs, paths) in enumerate(loader):
            images = images.to(device)

            outputs = model(images)  # [B, num_classes]
            if outputs.ndim != 2:
                outputs = outputs.view(outputs.shape[0], -1)

            class_indices = outputs.argmax(dim=1)  # [B]

            model.zero_grad()
            one_hot = torch.zeros_like(outputs)
            for i in range(outputs.shape[0]):
                one_hot[i, class_indices[i]] = 1.0
            outputs.backward(gradient=one_hot, retain_graph=False)

            acts = cam_hook.activations
            grads = cam_hook.gradients

            if acts is None or (args.cam_type in ["gradcam", "gradcampp"] and grads is None):
                print("[ERROR] Activations or gradients not captured.")
                break

            if args.cam_type == "gradcam":
                cams = compute_gradcam(acts, grads)
            elif args.cam_type == "gradcampp":
                cams = compute_gradcampp(acts, grads)
            elif args.cam_type == "eigencam":
                cams = compute_eigencam(acts)
            else:
                raise ValueError(f"Unknown cam_type: {args.cam_type}")

            cams_np = cams.cpu().numpy()

            for i in range(len(paths)):
                cam_map = cams_np[i]
                pil_img = resized_imgs[i]
                out_img = overlay_cam_on_image(pil_img, cam_map)

                rel_path = os.path.relpath(paths[i], args.data_dir)
                rel_root, _ = os.path.splitext(rel_path)
                out_path = os.path.join(args.save_dir, rel_root + "_cam.png")

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                out_img.save(out_path)

            print(f"[BATCH {batch_idx+1}/{len(loader)}] processed.")

    cam_hook.remove()
    print("[INFO] CAM generation finished.")


if __name__ == "__main__":
    main()
