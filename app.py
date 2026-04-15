import os, io, json, base64
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from flask import Flask, render_template, request, jsonify

# ======================================================
# MODEL DEFINITIONS (EXACTLY AS TRAINED)
# ======================================================

class LearnedGroupConv(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1, tau=0.005):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False
        )
        self.tau = tau
        self.scores = nn.Parameter(torch.randn(in_channels))

    def forward(self, x):
        out = self.depthwise(x)
        mask = (self.scores.abs() > self.tau).float().view(1, -1, 1, 1)
        return out * mask


class ResidualBlockLG(nn.Module):
    def __init__(self, channels, dropout_p=0.1):
        super().__init__()
        self.lgconv = LearnedGroupConv(channels)
        self.pw_conv = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_p)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.lgconv(x)
        out = self.pw_conv(out)
        out = self.bn(out)
        out = self.dropout(out)
        return self.relu(out + identity)


class DynLeafNet(nn.Module):
    def __init__(self, num_classes, channels=256, dropout_p=0.1, num_blocks=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, 1, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList([
            ResidualBlockLG(channels, dropout_p)
            for _ in range(num_blocks)
        ])

        self.exit_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)) for _ in range(num_blocks)
        ])
        self.exit_fcs = nn.ModuleList([
            nn.Linear(channels, num_classes) for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.relu(self.conv1(x))
        for block in self.blocks:
            x = block(x)
        pooled = self.exit_pools[-1](x).view(x.size(0), -1)
        return self.exit_fcs[-1](pooled)

# ======================================================
# GRAD-CAM++
# ======================================================

class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, module, inp, out):
        self.activations = out

    def _bwd(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx]
        score.backward(retain_graph=True)

        A = self.activations[0]
        dA = self.gradients[0]
        dA2, dA3 = dA**2, dA**3
        eps = 1e-8

        alpha = (dA2 / (2*dA2 + A*dA3 + eps)).sum(dim=(1,2))
        weights = alpha * F.relu(dA).sum(dim=(1,2))

        cam = (weights.view(-1,1,1) * A).sum(0)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max() + eps)
        return cam.detach().cpu().numpy()

# ======================================================
# MODEL LOADING (ROBUST: BOTH CONFIG FORMATS)
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_class_names(path):
    with open(path) as f:
        data = json.load(f)
    if "classes" in data:
        return [v for k, v in sorted(data["classes"].items(), key=lambda x: int(x[0]))]
    return [v for k, v in sorted(data.items(), key=lambda x: int(x[0]))]


def load_model_bundle(folder_name, file_prefix):
    root = os.path.join("models", folder_name)

    config_path = os.path.join(root, f"training_config_{file_prefix}.json")
    class_path  = os.path.join(root, f"class_names_{file_prefix}.json")
    weight_path = os.path.join(root, f"dynleafnet_{file_prefix}_best.pth")

    for p in [config_path, class_path, weight_path]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    with open(config_path) as f:
        cfg = json.load(f)

    # ---- Handle BOTH config formats ----
    if "model" in cfg and "training" in cfg:
        # PlantVillage (nested)
        num_classes = cfg["model"]["num_classes"]
        channels    = cfg["model"]["channels"]
        num_blocks  = cfg["model"]["num_blocks"]
        dropout_p   = cfg["model"]["dropout_p"]
        img_size    = cfg["training"]["img_size"]
    else:
        # Rice (flat)
        num_classes = cfg["num_classes"]
        channels    = cfg["channels"]
        num_blocks  = cfg["num_blocks"]
        dropout_p   = cfg["dropout_p"]
        img_size    = cfg["img_size"]

    model = DynLeafNet(
        num_classes=num_classes,
        channels=channels,
        dropout_p=dropout_p,
        num_blocks=num_blocks
    ).to(device)

    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    cam = GradCAMPlusPlus(model, model.blocks[-1].lgconv.depthwise)

    return {
        "model": model,
        "class_names": load_class_names(class_path),
        "transform": transform,
        "cam": cam
    }

# API name → (folder name, file prefix)
MODELS = {
    "plantvillage": load_model_bundle("plantvillage", "plantvillage"),
    "rice": load_model_bundle("rice-leaf", "rice")
}

# ======================================================
# FLASK APP
# ======================================================

app = Flask(__name__)

def np_to_b64(img_np):
    img = Image.fromarray(img_np.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    dataset = request.form.get("dataset", "plantvillage")
    if dataset not in MODELS:
        return jsonify({"error": "Invalid dataset"}), 400

    bundle = MODELS[dataset]

    img = Image.open(request.files["file"]).convert("RGB")
    orig_np = np.array(img)
    orig_size = img.size

    x = bundle["transform"](img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = bundle["model"](x)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(1)

    cls_idx = pred.item()
    disease = bundle["class_names"][cls_idx]
    confidence = float(conf.item())

    cam = bundle["cam"](x, cls_idx)
    cam = cv2.resize(cam, orig_size)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_np, 0.6, heat, 0.4, 0)

    return jsonify({
        "dataset": dataset,
        "disease": disease,
        "confidence": f"{confidence*100:.2f}%",
        "original": np_to_b64(orig_np),
        "overlay": np_to_b64(overlay)
    })

if __name__ == "__main__":
    app.run(debug=True)
