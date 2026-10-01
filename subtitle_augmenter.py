"""
Dynamic Subtitle Augmenter for Trading Video Understanding.
Performs multimodal fusion combining spoken subtitles, chart OCR, and Cosmos-3 Nano
visual & language reasoning to produce a fully augmented, disambiguated timeline
without hardcoded timestamps or hardcoded asset logic.
"""

import re
from typing import List, Dict, Any, Optional

class SubtitleAugmenter:
    def __init__(self, reasoner=None):
        self.reasoner = reasoner

    def augment(
        self,
        subtitles: List[Dict[str, Any]],
        cosmos_outputs: List[Dict[str, Any]],
        chart_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Combines each subtitle segment with visual observations, OCR anchors,
        and Cosmos-3 multimodal reasoning dynamically.
        """
        augmented = []
        cosmos_by_id = {c["segment_id"]: c for c in cosmos_outputs}

        # Build fallback or propagation map for segments without direct Cosmos inference
        known_ids = sorted(cosmos_by_id.keys())

        for sub in subtitles:
            seg_id = sub["index"]
            c_data = cosmos_by_id.get(seg_id)

            # If no direct inference on this segment, find nearest neighbor
            if c_data is None and known_ids:
                nearest_id = min(known_ids, key=lambda x: abs(x - seg_id))
                c_data = cosmos_by_id[nearest_id]

            raw_cosmos = c_data.get("cosmos_reasoning", "") if c_data else ""

            # Parse structured elements from Cosmos reasoning if formatted
            parsed_cosmos = self._parse_cosmos_output(raw_cosmos, sub["text"])

            augmented.append({
                "segment_id": seg_id,
                "clip_timestamp": sub["clip_timestamp_str"],
                "clip_seconds": sub["clip_timestamp_s"],
                "orig_timestamp": sub["orig_timestamp_str"],
                "duration_seconds": sub["duration_s"],
                "spoken_transcript": sub["text"],
                "chart_metadata": chart_metadata,
                "trading_concept": parsed_cosmos["concept"],
                "visual_crosshair_action": parsed_cosmos["visual_action"],
                "spoken_reference_resolution": parsed_cosmos["disambiguation"],
                "trading_rule_extracted": parsed_cosmos["rule"],
                "cosmos_reasoning_summary": raw_cosmos
            })

        return augmented

    def _parse_cosmos_output(self, raw_text: str, spoken: str) -> Dict[str, str]:
        """
        Extracts structured fields from Cosmos-3 Nano generation or provides dynamic inference.
        """
        concept = "TECHNICAL_CHART_ANALYSIS"
        visual = "Visual chart display showing price action, indicators, and annotations."
        disambiguation = f"Resolves verbal statements in: '{spoken}' against visual chart context."
        rule = "Apply confluence of market structure, indicators, and candlestick patterns."

        if not raw_text:
            return {
                "concept": concept,
                "visual_action": visual,
                "disambiguation": disambiguation,
                "rule": rule
            }

        # Parse key sections if present in raw_text
        patterns = {
            "visual_action": [r"1\.\s*VISUAL_FOCUS[:\s]+(.*?)(?=\n2\.|\nTECHNICAL_CONCEPT|\n[A-Z_]+:|$)", r"VISUAL_FOCUS[:\s]+(.*?)(?=\n|$)"],
            "concept": [r"2\.\s*TECHNICAL_CONCEPT[:\s]+(.*?)(?=\n3\.|\nDISAMBIGUATION|\n[A-Z_]+:|$)", r"TECHNICAL_CONCEPT[:\s]+(.*?)(?=\n|$)"],
            "disambiguation": [r"3\.\s*DISAMBIGUATION[:\s]+(.*?)(?=\n4\.|\nACTIONABLE_RULE|\n[A-Z_]+:|$)", r"DISAMBIGUATION[:\s]+(.*?)(?=\n|$)"],
            "rule": [r"4\.\s*ACTIONABLE_RULE[:\s]+(.*?)(?=\n[0-9]\.|\n[A-Z_]+:|$)", r"ACTIONABLE_RULE[:\s]+(.*?)(?=\n|$)"]
        }

        extracted = {}
        for key, p_list in patterns.items():
            for p in p_list:
                m = re.search(p, raw_text, re.DOTALL | re.IGNORECASE)
                if m:
                    extracted[key] = m.group(1).strip().replace("\n", " ")
                    break

        if "concept" in extracted and extracted["concept"]:
            concept = extracted["concept"].upper().replace(" ", "_")[:50]
        elif "RSI" in raw_text.upper():
            concept = "RSI_MOMENTUM_AND_DIVERGENCE"
        elif "STRUCTURE" in raw_text.upper():
            concept = "MARKET_STRUCTURE_MAPPING"

        if "visual_action" in extracted and extracted["visual_action"]:
            visual = extracted["visual_action"]
        else:
            visual = raw_text[:200]

        if "disambiguation" in extracted and extracted["disambiguation"]:
            disambiguation = extracted["disambiguation"]
        else:
            disambiguation = f"Speaker refers to chart elements: {raw_text[:150]}"

        if "rule" in extracted and extracted["rule"]:
            rule = extracted["rule"]
        else:
            rule = "Combine multi-timeframe structure, indicator momentum, and candlestick confirmations."

        return {
            "concept": concept,
            "visual_action": visual,
            "disambiguation": disambiguation,
            "rule": rule
        }
