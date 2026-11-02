# -*- coding: utf-8 -*-
"""语料词频统计：从一堆文本里统计高频词，用于提炼专业术语表

用法:
    python wordfreq.py <文件或目录> [更多...] [--top 300]

结果保存到 D:\\whisper\\词频统计结果.txt（每行: 词<TAB>次数，按次数降序）
"""
import argparse
import re
from collections import Counter
from pathlib import Path

import jieba

EN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.#]{1,}")  # 英文词 / 标识符
CN_RE = re.compile(r"[\u4e00-\u9fff]+")                # 连续中文片段

STOPWORDS = set(
    """
的 了 是 在 我 你 他 她 它 我们 你们 他们 这 那 就 都 也 而 及 与 和 或 但 如果 因为 所以
一个 没有 什么 怎么 为什么 这个 那个 这些 那些 可以 应该 需要 进行 使用 通过 对于 关于 以及
时候 然后 这样 那样 就是 不是 还是 只是 一直 已经 现在 以后 之前 之后 的话 一下 一些 可能 大概
比较 非常 很 太 更 最 又 再 才 就 着 了 过 被 把 让 给 到 从 向 往 对 为 以 是 有 会 能 要 不
the a an is are was were be been being to of in on at for and or but not this that these those
it its they we you he she i with as by from have has had do does did will would can could should may might
""".split()
)

TEXT_EXTS = {
    ".txt", ".md", ".markdown", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".sh", ".bat",
    ".json", ".csv", ".log", ".yaml", ".yml", ".toml", ".ini", ".conf", ".html", ".xml",
}


def main():
    ap = argparse.ArgumentParser(description="语料词频统计")
    ap.add_argument("paths", nargs="+", help="文件或目录")
    ap.add_argument("--top", type=int, default=300)
    args = ap.parse_args()

    files = []
    for p in args.paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.suffix.lower() in TEXT_EXTS:
                    files.append(f)
        elif p.is_file():
            files.append(p)

    if not files:
        print("没有找到文本文件。")
        return

    counter = Counter()
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for w in EN_RE.findall(text):          # 英文词/标识符
            counter[w.lower()] += 1
        for seg in CN_RE.findall(text):        # 中文分词
            for w in jieba.cut(seg):
                w = w.strip()
                if len(w) >= 2 and w not in STOPWORDS:
                    counter[w] += 1

    top = counter.most_common(args.top)
    out = Path(r"D:\whisper\词频统计结果.txt")
    out.write_text("\n".join(f"{w}\t{c}" for w, c in top), encoding="utf-8")

    print(f"[处理文件 {len(files)} 个 | 去重词 {len(counter)} 个 | 结果已保存]")
    print(out)
    print("-" * 44)
    for w, c in top[:60]:
        print(f"{c:>6}  {w}")
    print(f"\n（完整 {len(top)} 个词见上方文件）")


if __name__ == "__main__":
    main()
