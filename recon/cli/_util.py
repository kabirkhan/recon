from radicli import Radicli

cli = Radicli()


def setup_cli():
    cli.run()


__all__ = ["cli", "setup_cli"]
