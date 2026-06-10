# -*- coding: utf-8 -*-
"""
加分项1：temperature 对比批量生成。
加载一次模型，固定起笔字与 max_new，在多个 temperature 下各生成多次。
用法：python gen_batch.py [--ckpt ckpt_best.pt] [--prompt 春] [--runs 3]
"""
import argparse

from dataset import load_vocab_json
from predict import generate_one, load_for_generate
from train import get_device

TEMPERATURES = [0.4, 0.8, 1.2]


def main() -> None:
    p = argparse.ArgumentParser(description="temperature 对比批量生成")
    p.add_argument("--ckpt", type=str, default="ckpt_best.pt")
    p.add_argument("--vocab", type=str, default="vocab.json")
    p.add_argument("--prompt", type=str, default="春")
    p.add_argument("--max_new", type=int, default=120)
    p.add_argument("--runs", type=int, default=3, help="每个 temperature 生成几次")
    args = p.parse_args()

    stoi, itos, _ = load_vocab_json(args.vocab)
    device = get_device()
    model, _hp = load_for_generate(args.ckpt, device)
    print(f"模型: {args.ckpt}  起笔: {args.prompt}  max_new: {args.max_new}\n")

    for t in TEMPERATURES:
        print(f"========== temperature = {t} ==========")
        for r in range(1, args.runs + 1):
            text = generate_one(
                model, stoi, itos, device,
                prompt=args.prompt, max_new=args.max_new,
                temperature=t, stop_newline=True,
            )
            print(f"[t={t} 第{r}次] {text}")
        print()


if __name__ == "__main__":
    main()
