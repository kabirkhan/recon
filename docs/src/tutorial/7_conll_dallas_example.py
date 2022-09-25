from recon import Dataset
from recon.types import Example, Span
from recon.operations.core import op_registry

from recon.operations.tokenization import add_tokens

from prodigy.components.printers import pretty_print


def main():
    person_example = Example(text="My friend is named Dallas.", spans=[Span(text="Dallas", start=19, end=25, label="PERSON")])
    gpe_example = Example(text="Dallas is a city in Texas.", spans=[Span(text="Dallas", start=0, end=6, label="GPE")])

    ds = Dataset("DallasExamples", [person_example, gpe_example])
    ds.apply_("recon.add_tokens.v1")

    ds.data[0].show()
    ds.data[1].show()

    pretty_print([ds.data[0].dict(), ds.data[1].dict()], views=["spans"])


if __name__ == "__main__":
    main()
