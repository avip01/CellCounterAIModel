"""Thin CLI entry points for training, inference, evaluation."""
import argparse
from src.train import train


def main():
    parser = argparse.ArgumentParser(description="Cell Counter CLI")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("train")
    sub.add_parser("infer")
    sub.add_parser("evaluate")

    args = parser.parse_args()
    if args.command == "train":
        train()
    else:
        print(f"Command {args.command} not yet implemented")


if __name__ == "__main__":
    main()
