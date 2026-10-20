import torch
import gradio as gr
from torch import nn
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class SRModel(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x):
        return (x + self.net(x)).clamp(0, 1)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SRModel().to(DEVICE)
model.load_state_dict(torch.load("sr_model.pth", map_location=DEVICE))
model.eval()

to_tensor = T.ToTensor()
to_image = T.ToPILImage()
downsample = T.Resize((16, 16), interpolation=InterpolationMode.BICUBIC)
upsample = T.Resize((32, 32), interpolation=InterpolationMode.BICUBIC)


def enhance(image):
    if image is None:
        raise gr.Error("Please upload an image first.")
    image = image.convert("RGB")
    low_resolution = downsample(image)
    model_input_image = upsample(low_resolution)
    model_input = to_tensor(model_input_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = model(model_input)[0].cpu()
    return to_image(prediction)


demo = gr.Interface(
    fn=enhance,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Image(type="pil", label="Model output (32 × 32)"),
    title="Tiny Super-Resolution Demo",
    description=(
        "Educational prototype trained on CIFAR-10 images at 32 × 32 pixels. "
        "Uploads are reduced to 16 × 16, enlarged to 32 × 32, then processed. "
        "Output is always 32 × 32 pixels; this is not a general-purpose photo enhancer."
    ),
)

if __name__ == "__main__":
    demo.launch()
