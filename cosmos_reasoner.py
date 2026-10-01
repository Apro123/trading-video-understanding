"""
Cosmos-3 Nano Multimodal Video Reasoning Engine for Trading Analysis.
Loads NVIDIA Cosmos 3 Nano (via Cosmos3OmniForConditionalGeneration & AutoProcessor)
to perform temporal reasoning over trading chart video clips and align visual
chart events with spoken subtitles.
"""

import os
import cv2
import json
import torch
from PIL import Image
from typing import List, Dict, Any, Optional

DEFAULT_COSMOS3_PATH = "/mnt/edrive/huggingface/hub/models--nvidia--Cosmos3-Nano/snapshots/e59a53c25979a090fa8706c9acc0c254a6e89b92"

class CosmosTradingReasoner:
    def __init__(
        self,
        model_path: str = DEFAULT_COSMOS3_PATH,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16
    ):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None
        self._is_loaded = False

    def load_model(self):
        """Loads Cosmos-3 Nano on GPU."""
        if self._is_loaded:
            return

        print(f"[CosmosTradingReasoner] Loading NVIDIA Cosmos 3 Nano from {self.model_path}...")
        from transformers import AutoProcessor
        from transformers.models.cosmos3_omni.modeling_cosmos3_omni import Cosmos3OmniForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self.model = Cosmos3OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            local_files_only=True
        )
        self.model.eval()
        self._is_loaded = True
        print(f"[CosmosTradingReasoner] Cosmos 3 Nano successfully loaded on {self.device}.")

    def extract_segment_frames(
        self,
        video_path: str,
        start_s: float,
        end_s: float,
        num_frames: int = 4,
        max_size: tuple = (1280, 720)
    ) -> List[Image.Image]:
        """Extracts evenly spaced frames from video between start_s and end_s."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        duration = max(0.5, end_s - start_s)
        step = duration / max(1, num_frames)
        timestamps = [start_s + i * step for i in range(num_frames)]

        frames = []
        for ts in timestamps:
            frame_idx = min(int(ts * fps), total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                frames.append(img)
        cap.release()
        return frames

    def reason_over_segment(
        self,
        video_path: str,
        start_s: float,
        end_s: float,
        spoken_text: str,
        segment_id: int = 0
    ) -> Dict[str, Any]:
        """
        Runs Cosmos 3 Nano multimodal reasoning on a specific temporal segment of the trading clip.
        """
        self.load_model()
        frames = self.extract_segment_frames(video_path, start_s, end_s, num_frames=4)
        if not frames:
            return {"error": "No frames extracted"}

        system_prompt = (
            "You are NVIDIA Cosmos 3 Nano, an expert multimodal foundation model specializing in "
            "technical financial chart analysis, market structure reasoning, and video understanding. "
            "Your task is to analyze video frames from a trading tutorial and disambiguate vague spoken language "
            "by connecting it to exact visual chart elements: candlestick patterns, swing highs, swing lows, "
            "break of structure (BOS), support/resistance levels, trendlines, and buy zones."
        )

        user_prompt = f"""Video segment time: {start_s:.1f}s to {end_s:.1f}s.
Spoken transcript: "{spoken_text}"

Analyze these chart frames and the spoken words to provide:
1. VISUAL_FOCUS: What specific candle, price level, drawing tool, or chart area is being pointed to or drawn?
2. MARKET_STRUCTURE: What is the current trend state (Bearish, Bullish, Consolidation, Reversal)? Identify any swing high, swing low, or breakout.
3. SPOKEN_DISAMBIGUATION: What does the speaker specifically mean when saying words like "here", "this candle", "1927", "break above", or "buy zone"?
4. TRADING_RULE: What actionable trading rule or strategy guideline is demonstrated in this moment?"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": 1.0},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], videos=[frames], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=384,
                temperature=0.2,
                do_sample=True,
                repetition_penalty=1.05
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        raw_reasoning = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        return {
            "segment_id": segment_id,
            "start_s": start_s,
            "end_s": end_s,
            "spoken_text": spoken_text,
            "cosmos_reasoning": raw_reasoning
        }

if __name__ == "__main__":
    print("[CosmosTradingReasoner] Module loaded.")
