"""Upload a trained checkpoint to the Hugging Face Hub.

Streamlit Cloud can't hold a 500MB checkpoint in git, so the app downloads
weights from the Hub at startup instead.

    uv run huggingface-cli login
    uv run scripts/push_to_hub.py --ckpt checkpoints/cda_reg --repo yourname/fairness-gpt2-qqp
"""

import argparse

from huggingface_hub import HfApi, create_repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Local checkpoint directory")
    ap.add_argument("--repo", required=True, help="e.g. yourname/fairness-gpt2-qqp")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    create_repo(args.repo, private=args.private, exist_ok=True)
    HfApi().upload_folder(
        folder_path=args.ckpt,
        repo_id=args.repo,
        commit_message="Upload fairness-aware GPT-2 paraphrase checkpoint",
    )
    print(f"Pushed {args.ckpt} -> https://huggingface.co/{args.repo}")
    print(f'Now add this to Streamlit secrets:\n\nMODEL_REPO = "{args.repo}"')


if __name__ == "__main__":
    main()
