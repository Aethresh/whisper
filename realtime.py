# -*- coding: utf-8 -*-
"""实时语音转文字（准实时：说完一句、停顿约 1 秒即出字）

依赖: faster-whisper + sounddevice（已装好）
用法:
    python realtime.py [模型名]     # 模型名默认 medium
退出:
    Ctrl + C

输出:
    1) 终端实时打印每一句
    2) 追加写入 D:\\whisper\\dictation.md（带时间戳，可直接当笔记）
"""
import sys
from datetime import datetime

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from opencc import OpenCC

cc = OpenCC("t2s")  # 繁体→简体

# ── 可调参数 ──────────────────────────────────────────
SR = 16000                # 采样率（whisper 内部用 16k，直接用）
CHUNK_SEC = 0.4           # 每次读取 0.4 秒音频
SILENCE_SEC = 1.0         # 停顿超过这个秒数，认为一句话说完
SPEECH_THRESHOLD = 0.012  # 音量阈值，越大越不灵敏（环境吵就调大，比如 0.02）
MIN_UTTER_SEC = 0.6       # 短于此秒数的片段忽略（过滤咳嗽/杂音）
OUTPUT_FILE = r"D:\whisper\dictation.md"
# 专业术语表（按需增删，帮助识别你的技术词汇）
TERMS = "PyTorch CUDA 神经网络 注意力机制 Transformer 大模型 微调 推理 部署 向量数据库 RAG 卷积 反向传播 激活函数 优化器"
# ──────────────────────────────────────────────────────

model_name = sys.argv[1] if len(sys.argv) > 1 else "medium"

print(f"[加载模型] {model_name} @ cuda ...")
model = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
print("[就绪] 开始实时转写，直接说话即可，Ctrl+C 退出。\n", flush=True)


def rms(a: np.ndarray) -> float:
    """音频片段的均方根（近似音量）"""
    return float(np.sqrt(np.mean(np.square(a))))


def save(text: str) -> None:
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(f"- {datetime.now().strftime('%H:%M:%S')}  {text}\n")


buf: list = []      # 当前一句话累积的音频
silent = 0.0        # 连续静音时长
speaking = False

try:
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
        while True:
            chunk, _ = stream.read(int(SR * CHUNK_SEC))
            chunk = chunk[:, 0]          # 压成单声道一维
            vol = rms(chunk)

            if vol > SPEECH_THRESHOLD:
                buf.append(chunk)
                speaking = True
                silent = 0.0
            elif speaking:
                buf.append(chunk)        # 保留句尾渐弱部分
                silent += CHUNK_SEC
                if silent >= SILENCE_SEC:
                    audio = np.concatenate(buf)
                    buf = []
                    speaking = False
                    silent = 0.0
                    if len(audio) / SR < MIN_UTTER_SEC:
                        continue
                    try:
                        segments, _ = model.transcribe(
                            audio,
                            language="zh",                      # 强制中文，避免误判成英/俄
                            vad_filter=True,                    # Silero 过滤静音/杂音，防幻觉
                            condition_on_previous_text=False,   # 防重复
                            initial_prompt=f"以下是简体中文普通话的句子。专业术语：{TERMS}",  # 引导简体+术语
                        )
                        text = cc.convert("".join(s.text for s in segments).strip())  # 繁转简
                        if text:
                            print(f"→ {text}", flush=True)
                            save(text)
                    except Exception as e:  # 单句出错不中断整体
                        print(f"[转写出错] {e}")
except KeyboardInterrupt:
    print("\n[已退出]")
