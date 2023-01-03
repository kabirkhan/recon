from recon import Dataset
from recon.operations.tokenization import *  # noqa
from recon.types import Example, Span


def main():
    person_example = Example(
        text="My friend is named Dallas.",
        spans=[Span(text="Dallas", start=19, end=25, label="PER")],
    )
    gpe_example = Example(
        text="Dallas is a city in Texas.",
        spans=[Span(text="Dallas", start=0, end=6, label="LOC")],
    )

    ds = Dataset("DallasExamples", [person_example, gpe_example])
    ds.apply_("recon.add_tokens.v1")

    ds.data[0].show()
    ds.data[1].show()


if __name__ == "__main__":
    main()
