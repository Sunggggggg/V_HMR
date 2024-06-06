import cv2
import torch.nn as nn
import clip
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel

# generate caption
gen_kwargs = {
    "min_length": 10, 
    "max_length": 30, 
    "num_beams": 8,
}

class CaptionEmb(nn.Module):
    def __init__(self, ) :
        super().__init__()
        # Video captioning
        self.image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning")

        # Clip
        self.clip = clip.load("ViT-B/32")

    def video_caption(self, seq_path):
        """
        seq_path : list 16
        """
        frames = []
        for path in seq_path :
            img = cv2.imread(path)
            H, W = img.shape[:2]
            H, W = H//4, W//4
            img = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
            frames.append(img)
        
        pixel_values = self.image_processor(frames, return_tensors="pt").pixel_values # [B, N, 3, H, W]
        tokens = self.model.generate(pixel_values, **gen_kwargs)
        caption = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]

        del frames
        return caption

    def forward(self, seq_path):
        """
        joint_2d : [B, T, J, 2]
        seq_path : 
        """
        caption = self.video_caption(seq_path) # [B, dim]

        # Text embedding
        clip_token = self.clip.tokenize(caption)
        text_emb = self.clip.encode_text(clip_token)

        return text_emb