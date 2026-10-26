"""
Integration test script for YOLO-World Large API Service.
Tests /health, /set_classes, /detect with a real chart frame, and /detect/visualize.
"""

import sys
import time
from pathlib import Path
try:
    from yolo_world_service.client import YoloWorldClient
except (ImportError, ModuleNotFoundError):
    from client import YoloWorldClient

def main():
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8005"
    print(f"=== Testing YOLO-World Large Service at {base_url} ===")
    
    client = YoloWorldClient(base_url=base_url)
    
    # 1. Health check
    print("\n1. Testing /health...")
    health = client.health()
    print("Health response:", health)
    assert health["status"] == "healthy"
    
    # 2. Set custom classes
    print("\n2. Testing /set_classes with trading chart vocabulary...")
    test_classes = [
        "candlestick",
        "price axis",
        "time axis",
        "indicator pane",
        "trendline",
        "support level",
        "resistance level",
        "breakout",
        "text label"
    ]
    set_res = client.set_classes(test_classes)
    print("Set classes response:", set_res)
    assert set_res["status"] == "success"
    
    # 3. Detect objects in sample chart frame
    test_image_path = Path("sample_frames/frame_000s.jpg")
    if not test_image_path.exists():
        test_image_path = Path("/workspace/sample_frames/frame_000s.jpg")
        
    print(f"\n3. Testing /detect on {test_image_path}...")
    if not test_image_path.exists():
        print(f"Sample frame {test_image_path} not found, generating dummy test image...")
        from PIL import Image
        img = Image.new("RGB", (640, 640), color=(30, 30, 30))
        img.save("dummy_test.jpg")
        test_image_path = Path("dummy_test.jpg")
        
    detect_res = client.detect(
        image=test_image_path,
        classes=["candlestick", "price axis", "indicator", "chart", "text"],
        confidence=0.15
    )
    print(f"Inference latency: {detect_res['inference_time_ms']} ms")
    print(f"Total detections: {len(detect_res['detections'])}")
    for i, det in enumerate(detect_res['detections'][:5]):
        print(f"  [{i+1}] Class: {det['class_name']} | Conf: {det['confidence']} | Normalized Box: {det['normalized_box']}")
        
    # 4. Test visualization endpoint
    print("\n4. Testing /detect/visualize endpoint...")
    out_vis_path = Path("test_yolo_vis.jpg")
    saved_path = client.detect_and_save_visual(
        image=test_image_path,
        output_path=out_vis_path,
        classes=["candlestick", "price axis", "indicator", "chart", "text"],
        confidence=0.15
    )
    print(f"Visualization saved to: {saved_path} (size: {saved_path.stat().st_size} bytes)")
    assert saved_path.stat().st_size > 0
    
    print("\n✅ ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
