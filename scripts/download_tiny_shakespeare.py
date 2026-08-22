"""Download the canonical Tiny Shakespeare corpus used by Karpathy's char-rnn."""

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "tiny_shakespeare.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="redownload the corpus even if the output file already exists",
    )
    args = parser.parse_args()

    output = DEFAULT_OUTPUT
    if output.exists() and not args.force:
        print(f"Corpus already exists at {output}")
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=30) as response:
        corpus = response.read()

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary_file:
            temporary_file.write(corpus)
            temporary_path = temporary_file.name
        os.replace(temporary_path, output)
    finally:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    print(f"Downloaded {len(corpus):,} bytes to {output}")


if __name__ == "__main__":
    main()
