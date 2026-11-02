# -*- coding: utf-8 -*-
"""faster-whisper 批量文件转写（支持多种音视频格式）

用法:
    python transcribe.py <文件或目录> [更多...] [选项]

示例:
    python transcribe.py 会议.mp3
    python transcribe.py a.mp4 b.m4a
    python transcribe.py D:\\录音目录
    python transcribe.py 文件.mp4 --language auto     # 自动检测语言

选项:
    --model MODEL       模型，默认 large-v3（可选 tiny/base/small/medium/turbo/large-v3）
    --language LANG     语言，默认 zh；填 auto 自动检测，或 en/ja 等

说明:
    - 支持 mp3/wav/flac/m4a/aac/ogg/opus 等音频，mp4/mkv/avi/mov/webm 等视频（PyAV 解码）
    - 每个文件转写结果保存为「原名.后缀.txt」
"""
import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel
from opencc import OpenCC

cc = OpenCC("t2s")  # 繁体→简体

# 专业术语表（按需增删，帮助识别你的技术词汇）
TERMS = "PyTorch CUDA 神经网络 注意力机制 Transformer 大模型 微调 推理 部署 向量数据库 RAG 卷积 反向传播 激活函数 优化器 梯度 训练 数据集 接口 API 模型 参数"
# 转写结果统一输出目录
OUTPUT_DIR = Path(r"D:\whisper\转写结果")

EXTS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma", ".amr",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".wmv", ".flv", ".ts",
    ".mpg", ".mpeg", ".3gp", ".rmvb",
}


def collect(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in EXTS:
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            print(f"[跳过] 路径不存在: {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description="faster-whisper 批量转写")
    ap.add_argument("paths", nargs="+", help="文件或目录")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--language", default="zh", help="zh / en / auto / ...")
    args = ap.parse_args()

    lang = None if args.language.lower() in ("auto", "none") else args.language
    files = collect(args.paths)
    if not files:
        print("没有找到可转写的音视频文件。")
        sys.exit(1)

    print(f"[加载模型] {args.model} @ cuda ...")
    model = WhisperModel(args.model, device="cuda", compute_type="int8_float16")

    ok = 0
    print(f"[共 {len(files)} 个文件]")
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {f.name}")
        try:
            init_prompt = f"以下是简体中文普通话的句子。专业术语：{TERMS}" if lang == "zh" else None
            segments, info = model.transcribe(
                str(f), language=lang, vad_filter=True, initial_prompt=init_prompt
            )
            text = "\n".join(s.text.strip() for s in segments).strip()
            if lang == "zh":
                text = cc.convert(text)  # 繁转简
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUTPUT_DIR / (f.name + ".txt")
            out.write_text(text, encoding="utf-8")
            ok += 1
            print(f"  [语言] {info.language} | [已保存] {out}")
            print(f"  [预览] {text[:100]}{'…' if len(text) > 100 else ''}")
        except Exception as e:
            print(f"  [出错] {e}")

    print(f"\n完成：{ok}/{len(files)} 个文件转写成功。")


if __name__ == "__main__":
    main()
