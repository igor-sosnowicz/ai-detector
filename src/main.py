"""Entry point to the application as a Typer CLI."""

import typer

from src.configuration import config

app = typer.Typer(help=f"CLI for {config.project_name}", no_args_is_help=True)


@app.command()
def main() -> None:
    """Print a greeting."""
    typer.echo(f"Hello, world! It is {config.project_name}.")


if __name__ == "__main__":
    app()
