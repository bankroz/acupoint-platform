"""下载 MediaPipe 模型文件"""
import os
import urllib.request

MODELS = {
    "pose_landmarker": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "holistic_landmarker": "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task",
}

def download_models(target_dir: str = "models"):
    os.makedirs(target_dir, exist_ok=True)
    for name, url in MODELS.items():
        path = os.path.join(target_dir, f"{name}.task")
        if os.path.exists(path):
            print(f"[SKIP] {path} 已存在")
            continue
        print(f"[DOWNLOAD] {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
            size_mb = os.path.getsize(path) / (1024*1024)
            print(f"  OK ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  FAILED: {e}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    download_models()
