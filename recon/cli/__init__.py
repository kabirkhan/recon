import typer

from .stats import stats

app = typer.Typer(no_args_is_help=True)


commands = [stats]
for command in commands:
    app.command(no_args_is_help=True)(command)


@app.callback()
def main() -> None:
    """
    \b
     _____    ______    _____    ____    _   _ 
    |  __ \  |  ____|  / ____|  / __ \  | \ | |
    | |__) | | |__    | |      | |  | | |  \| |
    |  _  /  |  __|   | |      | |  | | | . ` |
    | | \ \  | |____  | |____  | |__| | | |\  |
    |_|  \_\ |______|  \_____|  \____/  |_| \_|                                       
    """
    pass


if __name__ == "__main__":
    app()
