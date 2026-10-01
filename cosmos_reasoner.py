"""
Cosmos-3 Nano Unified Multimodal & Text Reasoning Engine.
Supports:
1. Multimodal video reasoning over sampled chart frames.
2. Direct text LLM reasoning (for dynamic subtitle augmentation and strategy synthesis).
3. Live VRAM monitoring tracking peak GPU memory usage on GB10 / DGX Spark.
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
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def get_vram_stats(self) -> Dict[str, float]:
        """Returns current and peak GPU VRAM usage in Gigabytes."""
        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}

        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        res = torch.cuda.memory_reserved() / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_res = torch.cuda.max_memory_reserved() / (1024 ** 3)

        return {
            "current_allocated_gb": round(alloc, 3),
            "current_reserved_gb": round(res, 3),
            "peak_allocated_gb": round(peak_alloc, 3),
            "peak_reserved_gb": round(peak_res, 3)
        }

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
        vram = self.get_vram_stats()
        print(f"[CosmosTradingReasoner] Cosmos 3 Nano loaded on {self.device}. VRAM: {vram['peak_allocated_gb']} GB allocated.")

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
            frame_idx = min(int(ts * fps), max(0, total_frames - 1))
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
        Multimodal video reasoning: examines frames and spoken words for a segment.
        """
        self.load_model()
        frames = self.extract_segment_frames(video_path, start_s, end_s, num_frames=4)
        if not frames:
            return {"error": "No frames extracted", "cosmos_reasoning": ""}

        system_prompt = (
            "You are NVIDIA Cosmos 3 Nano, an expert multimodal foundation model specializing in "
            "technical financial chart analysis, indicators (RSI, MA, MACD), market structure, "
            "and video understanding. Analyze the provided video frames alongside the spoken words."
        )

        user_prompt = f"""Video segment time: {start_s:.1f}s to {end_s:.1f}s.
Spoken transcript: "{spoken_text}"

Analyze the chart video frames and spoken words:
1. VISUAL_FOCUS: What indicators, candlesticks, cursor movements, drawings, or levels are shown?
2. TECHNICAL_CONCEPT: What trading concept is shown (e.g. RSI Divergence, Overbought/Oversold, Break of Structure, Swing Points, Support/Resistance)?
3. DISAMBIGUATION: Explain what the speaker specifically means by words like 'here', 'this indicator', '70', '30', or 'reversal'.
4. ACTIONABLE_RULE: What is the core rule or lesson communicated?"""

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
                max_new_tokens=350,
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

    def generate_text_reasoning(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2
    ) -> str:
        """
        Uses Cosmos 3 Nano as a pure text LLM for downstream synthesis and structured reasoning.
        """
        self.load_model()
        if system_prompt is None:
            system_prompt = (
                "You are an expert quantitative trading analyst and LLM reasoning engine. "
                "Synthesize financial trading lessons and technical chart observations into "
                "clear, precise, actionable trading rules."
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], padding=True, return_tensors="pt").to(self.device)

        with torch.no_grad():
            gen_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                repetition_penalty=1.05
            )

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

if __name__ == "__main__":
    print("[CosmosTradingReasoner] Module loaded.")
