"""
Arabic Sign Language Recognition — Gradio Demo
"""

# !pip install gradio -q

import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
    def forward(self, x):
        return self.pool(self.block(x) + self.shortcut(x))

class ArSLCNN_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = ConvBlock(3, 64); self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256); self.block4 = ConvBlock(256, 512)
        self.block5 = ConvBlock(512, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(1024, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.gap(self.block5(self.block4(self.block3(self.block2(self.block1(x)))))))

CHECKPOINT  = "/content/outputs_v2/best_model_v2.pth"
checkpoint  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
CLASS_NAMES = checkpoint["class_names"]
IMG_SIZE    = checkpoint["config"]["image_size"]

model = ArSLCNN_v2(num_classes=len(CLASS_NAMES)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"[INFO] Model loaded | {len(CLASS_NAMES)} classes | val_acc: {checkpoint['val_acc']*100:.2f}%")

ARABIC_NAMES = {
    "ALIF":"ا (Alif)","AYN":"ع (Ayn)","BAA":"ب (Baa)","DAD":"ض (Dad)",
    "DELL":"د (Dell)","DHAA":"ظ (Dhaa)","DHELL":"ذ (Dhell)","FAA":"ف (Faa)",
    "GHAYN":"غ (Ghayn)","HAA":"ه (Haa)","HAH":"ح (Hah)","JEEM":"ج (Jeem)",
    "KAF":"ك (Kaf)","KHAA":"خ (Khaa)","LAM":"ل (Lam)","MEEM":"م (Meem)",
    "NOON":"ن (Noon)","QAF":"ق (Qaf)","RAA":"ر (Raa)","SAD":"ص (Sad)",
    "SEEN":"س (Seen)","SHEEN":"ش (Sheen)","TA":"ت (Ta)","TAA":"ط (Taa)",
    "THA":"ث (Tha)","WAW":"و (Waw)","YAA":"ي (Yaa)","ZAY":"ز (Zay)",
}

infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def predict(image):
    if image is None:
        return "No image provided", {}
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert("RGB")
    tensor = infer_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top_class  = CLASS_NAMES[probs.argmax().item()]
    confidence = probs.max().item() * 100
    conf_dict  = {ARABIC_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i]): round(probs[i].item(), 4) for i in range(len(CLASS_NAMES))}
    return f"**{ARABIC_NAMES.get(top_class, top_class)}**\nConfidence: {confidence:.1f}%", conf_dict

with gr.Blocks(title="Arabic Sign Language Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤟 Arabic Sign Language Recognition\n**Milestone 3 | Pattern Recognition — PSUT**\n*Sultan Abdullah | Omar Khaled | Zain Sinokort*")
    with gr.Tabs():
        with gr.Tab("📁 Upload Image"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(label="Upload Hand Gesture", type="numpy")
                    btn = gr.Button("Recognize", variant="primary")
                with gr.Column():
                    lbl = gr.Markdown()
                    cnf = gr.Label(label="Top Probabilities (%)", num_top_classes=5)
            btn.click(predict, inputs=inp, outputs=[lbl, cnf])
        with gr.Tab("📷 Webcam"):
            with gr.Row():
                with gr.Column():
                    cam = gr.Image(label="Webcam — capture to recognize instantly",
                                  sources=["webcam"], type="numpy", mirror_webcam=True)
                with gr.Column():
                    lbl2 = gr.Markdown()
                    cnf2 = gr.Label(label="Top Probabilities (%)", num_top_classes=5)
            cam.change(predict, inputs=cam, outputs=[lbl2, cnf2])
    gr.Markdown("---\n**Model:** ArSLCNN v2 | 28 classes | Test Accuracy: 96.4%")

demo.launch(share=True, debug=False)
