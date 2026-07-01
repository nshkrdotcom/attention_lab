import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda version:", torch.version.cuda)
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Fix the environment before training.")


if __name__ == "__main__":
    main()

