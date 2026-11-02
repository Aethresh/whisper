# -*- coding: utf-8 -*-
"""麦克风自检：列出音频设备，并试着采集 0.5 秒音频。"""
import numpy as np
import sounddevice as sd

print("默认输入设备索引:", sd.default.device[0])
print(sd.query_devices())
print()

try:
    with sd.InputStream(samplerate=16000, channels=1, dtype="float32") as s:
        d, _ = s.read(8000)
    vol = float(np.sqrt(np.mean(np.square(d))))
    print(f"麦克风采集 OK：读到 {len(d)} 样本，当前音量 {vol:.4f}")
    print("（对着麦克风说话，若这个音量值明显变大，说明阈值判断能正常工作）")
except Exception as e:
    print("麦克风采集失败：", e)
