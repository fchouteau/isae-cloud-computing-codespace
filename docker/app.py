import time
from pathlib import Path
from typing import Annotated, Optional

import typer

app = typer.Typer()


@app.command()
def say_hello(name: str):
    typer.echo(f"Hello {name}")


@app.command()
def run_training(
    config: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path,
        typer.Option(
            dir_okay=True,
            writable=True,
            readable=True,
            resolve_path=True,
            file_okay=False,
        ),
    ],
):
    text = config.read_text()
    print(f"Config file contents: {text}")

    print(f"Running training in {output_dir}...")

    time.sleep(10)

    output_dir.mkdir(exist_ok=True,parents=True)

    with open(output_dir / "results.txt", "w") as f:
        f.write("Training successful !")


if __name__ == "__main__":
    app()
