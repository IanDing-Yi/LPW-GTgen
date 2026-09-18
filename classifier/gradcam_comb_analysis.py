#!/usr/bin/env python
"""
Grad-CAM interpretability analysis for the Combined-only (comb) ConvNeXtV2 experiment,
k=1 (train_size=1), split 1 (first random-eval split), evaluated on real test fragments.

Produces two groups of heatmaps:
  - correct/   Grad-CAM overlays for correctly classified real fragments
  - incorrect/ Grad-CAM overlays for misclassified real fragments (CAM w.r.t. the
               predicted (wrong) class, showing what drove the wrong decision)

Also writes a per-image CSV report and summary montage grids for each group to help
answer:
  Q1: Do correctly classified real fragments show attention on archaeologically
      diagnostic regions (rims, handles, decoration) rather than generic silhouette?
  Q2: Do misclassified fragments show attention on generic silhouette/background/color
      regions instead of diagnostic features?

Usage:
    python gradcam_comb_analysis.py
    python gradcam_comb_analysis.py --out_dir gradcam_out --device cuda
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage import io, transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pretrain_model import get_pretrain_model

CLASS_NAMES = ['bowl', 'butter_crock', 'canister', 'jug', 'flower_pot']

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(img_path, size=224):
    """Reproduce the Rescale->ToTensor->Normalize pipeline used at train/test time,
    while also keeping the resized RGB image (0-1 float) for overlay display."""
    image = io.imread(img_path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    image = transform.resize(image, (size, size), anti_aliasing=True)
    display_img = image.astype(np.float32)  # HxWx3, 0-1

    tensor = image.transpose((2, 0, 1)).astype(np.float32)
    tensor = torch.from_numpy(tensor)
    tensor = (tensor - torch.tensor(MEAN).view(3, 1, 1)) / torch.tensor(STD).view(3, 1, 1)
    return tensor, display_img


class GradCAM:
    """Grad-CAM hooked on the last ConvNeXtV2 stage (last spatial feature map
    before the backbone's own global pooling / classifier head)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=-1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        acts = self.activations[0]      # [C, H, W]
        grads = self.gradients[0]        # [C, H, W]
        weights = grads.mean(dim=(1, 2))  # [C]

        cam = torch.zeros(acts.shape[1:], dtype=acts.dtype, device=acts.device)
        for c in range(acts.shape[0]):
            cam += weights[c] * acts[c]
        cam = F.relu(cam)
        cam = cam.detach().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam, logits.detach()


def overlay_cam(display_img, cam, size=224, alpha=0.45):
    cam_resized = transform.resize(cam, (size, size), anti_aliasing=True)
    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = (1 - alpha) * display_img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay, cam_resized


def make_montage(records, title, out_path, max_items=24, ncols=6):
    records = records[:max_items]
    if not records:
        return
    nrows = int(np.ceil(len(records) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.4))
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, rec in zip(axes, records):
        ax.imshow(rec['overlay'])
        ax.set_title(f"T:{rec['true_name']} P:{rec['pred_name']}\nconf={rec['confidence']:.2f}", fontsize=8)
        ax.axis('off')
    for ax in axes[len(records):]:
        ax.axis('off')
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default='')
    parser.add_argument('--model_name', default='convnextv2_base')
    parser.add_argument('--model_path',
                         default='1_groundtruth_swap_convnextv2_base_gt_rand_convnextv2_base_comb.pth')
    parser.add_argument('--test_csv',
                         default=os.path.join(
                             '1_groundtruth_swap_convnextv2_base_gt_rand',
                             '1_groundtruth_swap_convnextv2_base_gt_rand_1',
                             'test.csv'))
    parser.add_argument('--out_dir', default='gradcam_comb_k1_split1')
    parser.add_argument('--nb_cls', type=int, default=5)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    print('Using device:', device)

    os.makedirs(args.out_dir, exist_ok=True)
    correct_dir = os.path.join(args.out_dir, 'correct')
    incorrect_dir = os.path.join(args.out_dir, 'incorrect')
    os.makedirs(correct_dir, exist_ok=True)
    os.makedirs(incorrect_dir, exist_ok=True)

    model = get_pretrain_model(args.model_name, args.nb_cls)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    target_layer = model.convnextv2_base_ft.stages[-1]
    gradcam = GradCAM(model, target_layer)

    df = pd.read_csv(args.test_csv, header=None, dtype=str)

    rows = []
    correct_records = []
    incorrect_records = []

    for idx in range(len(df)):
        rel_path = df.iloc[idx, 0]
        true_label = int(df.iloc[idx, 1])
        img_path = os.path.join(args.base_path, rel_path)

        tensor, display_img = load_image(img_path)
        input_tensor = tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(False)

        cam, logits = gradcam(input_tensor, class_idx=None)
        probs = torch.softmax(logits, dim=-1)[0]
        pred_label = int(torch.argmax(probs).item())
        confidence = float(probs[pred_label].item())
        is_correct = (pred_label == true_label)

        overlay, cam_resized = overlay_cam(display_img, cam)

        base_name = os.path.splitext(os.path.basename(rel_path))[0]
        out_name = f"{idx:03d}_{base_name}_true-{CLASS_NAMES[true_label]}_pred-{CLASS_NAMES[pred_label]}.png"
        out_subdir = correct_dir if is_correct else incorrect_dir
        out_path = os.path.join(out_subdir, out_name)
        plt.imsave(out_path, overlay)

        record = {
            'overlay': overlay,
            'true_name': CLASS_NAMES[true_label],
            'pred_name': CLASS_NAMES[pred_label],
            'confidence': confidence,
        }
        (correct_records if is_correct else incorrect_records).append(record)

        rows.append({
            'image_path': rel_path,
            'true_label': true_label,
            'true_name': CLASS_NAMES[true_label],
            'pred_label': pred_label,
            'pred_name': CLASS_NAMES[pred_label],
            'confidence': confidence,
            'correct': is_correct,
            'heatmap_path': out_path,
        })

        print(f"[{idx+1}/{len(df)}] {rel_path} true={CLASS_NAMES[true_label]} "
              f"pred={CLASS_NAMES[pred_label]} conf={confidence:.3f} correct={is_correct}")

    report_df = pd.DataFrame(rows)
    report_csv = os.path.join(args.out_dir, 'gradcam_report.csv')
    report_df.to_csv(report_csv, index=False)

    acc = report_df['correct'].mean()
    print(f"\nSplit-1 real-fragment test accuracy: {acc*100:.2f}% "
          f"({report_df['correct'].sum()}/{len(report_df)})")

    make_montage(correct_records,
                 f"Correctly classified (n={len(correct_records)}) - Grad-CAM (predicted class)",
                 os.path.join(args.out_dir, 'montage_correct.png'))
    make_montage(incorrect_records,
                 f"Misclassified (n={len(incorrect_records)}) - Grad-CAM (predicted class)",
                 os.path.join(args.out_dir, 'montage_incorrect.png'))

    print(f"\nSaved per-image heatmaps under: {correct_dir} and {incorrect_dir}")
    print(f"Saved summary montages: montage_correct.png, montage_incorrect.png in {args.out_dir}")
    print(f"Saved report: {report_csv}")


if __name__ == '__main__':
    main()
