# Checkpoints

Pretrained EMA weights are hosted in the following Hugging Face repositories:

- [`progresvit_192_240.pth.tar`](https://huggingface.co/NCPS/progresvit-deit-s-192-240-imagenet1k)
- [`progresvit_192_240_kd.pth.tar`](https://huggingface.co/NCPS/progresvit-deit-s-192-240-kd-imagenet1k)
- [`progresvit_160_384.pth.tar`](https://huggingface.co/NCPS/progresvit-deit-s-160-384-imagenet1k)
- [`progresvit_160_384_kd.pth.tar`](https://huggingface.co/NCPS/progresvit-deit-s-160-384-kd-imagenet1k)

Checkpoint weights are intentionally not tracked by Git.

From the repository root, download all four files into this directory with:

```bash
bash download_checkpoints.sh
```

Alternatively, download a file manually from the corresponding link above and place it in this directory.
