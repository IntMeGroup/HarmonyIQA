
import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

MODEL_PATH = "/mnt/data/xzt/LLM/internvl3_5"
INPUT_SIZE = 448

IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

query = (
    "Image harmonization aims to make the foreground consistent with the background. "
    "Please score the image harmonization quality from 0 to 100, where higher is better. <image> "
    "Reply: The image quality score is XX.XX"
)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize(
            (input_size, input_size),
            interpolation=InterpolationMode.BICUBIC
        ),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=2048,
                 hidden_dim2=128, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


def load_model(peft_dir, device=None):
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float32
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, use_fast=False
    )
    config = AutoConfig.from_pretrained(
        MODEL_PATH, trust_remote_code=True
    )

    img_context_token_id = tokenizer.convert_tokens_to_ids(
        IMG_CONTEXT_TOKEN
    )
    image_size = config.force_image_size or config.vision_config.image_size
    patch_size = config.vision_config.patch_size
    num_image_token = int(
        (image_size // patch_size) ** 2
        * (config.downsample_ratio ** 2)
    )

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    # Load vision and language LoRA adapters
    model.vision_model = PeftModel.from_pretrained(
        model.vision_model,
        os.path.join(peft_dir, "vision_lora")
    )
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        os.path.join(peft_dir, "llm_lora")
    )

    model.img_context_token_id = img_context_token_id
    model = model.to(device).eval()

    hidden_size = getattr(config, "hidden_size", 4096)
    mlp = MLPRegressor(
        input_dim=hidden_size,
        hidden_dim=hidden_size // 2,
        hidden_dim2=128
    )
    mlp.load_state_dict(
        torch.load(
            os.path.join(peft_dir, "mlp_head.pth"),
            map_location="cpu",
            weights_only=True
        )
    )
    mlp = mlp.to(device).eval()

    return model, mlp, tokenizer, num_image_token, device, dtype


@torch.inference_mode()
def infer_single(image_path, model, mlp, tokenizer,
                 num_image_token, device, dtype):
    transform = build_transform(INPUT_SIZE)

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(
        device=device, dtype=dtype
    )

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * num_image_token
        + IMG_END_TOKEN
    )
    prompt = query.replace("<image>", image_tokens, 1)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    context_count = (
        inputs["input_ids"] == model.img_context_token_id
    ).sum().item()
    if context_count == 0:
        raise ValueError(
            "No <IMG_CONTEXT> tokens found. Check the query and tokenizer."
        )

    image_flags = torch.ones(
        (1, pixel_values.size(0)),
        dtype=torch.long,
        device=device
    )

    outputs = model(
        pixel_values=pixel_values,
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_flags=image_flags,
        output_hidden_states=True
    )

    # Match the feature extraction used during training
    feats = outputs.hidden_states[-1][:, -6, :].float()
    score = mlp(feats).abs().squeeze().item() * 100

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--peft_dir", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model, mlp, tokenizer, num_image_token, device, dtype = load_model(
        args.peft_dir, args.device
    )

    score = infer_single(
        args.image, model, mlp, tokenizer,
        num_image_token, device, dtype
    )

    print(f"Image harmonization quality score: {score:.2f}")
