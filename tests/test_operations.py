import pytest
from recon.dataset import Dataset
from recon.operations import Operation, operation, registry
from recon.types import Example, Span


@pytest.fixture()
def ds():
    ds = Dataset(
        name="test",
        data=[
            Example(
                text="this is a test example with something else",
                spans=[Span(text="something", start=28, end=37, label="TEST_ENTITY")],
            )
        ],
    )
    ds.apply_("recon.v1.add_tokens")

    return ds


def test_operation_init():
    @operation("test_operation")
    def operation_test(example):
        return example

    assert "test_operation" in registry.operations
    assert isinstance(registry.operations.get("test_operation"), Operation)


def test_change_operation(ds):
    @operation("change_annotation")
    def operation_test(example):
        example.spans[0].text = "something else"
        example.spans[0].end = 42

        return example

    assert "change_annotation" in registry.operations
    assert isinstance(registry.operations.get("change_annotation"), Operation)

    assert len(ds.operations) == 1
    ds.apply_("change_annotation")
    assert len(ds.operations) == 2

    assert ds.operations[1].name == "change_annotation"
    assert len(ds.example_store) == 3

    assert len(ds) == 1


def test_add_operation(ds):
    @operation("add_and_change_example")
    def operation_test(example):
        new_example = Example(text="this is a test", spans=[])

        example.spans[0].text = "something else"
        example.spans[0].end = 42

        return [new_example, example]

    assert "add_and_change_example" in registry.operations
    assert isinstance(registry.operations.get("add_and_change_example"), Operation)

    assert len(ds.operations) == 1
    ds.apply_("add_and_change_example")
    assert len(ds.operations) == 2

    assert ds.operations[1].name == "add_and_change_example"
    assert len(ds.example_store) == 4

    assert len(ds) == 2


def test_remove_operation(ds):
    @operation("remove_example")
    def operation_test(example):
        return None

    assert "remove_example" in registry.operations
    assert isinstance(registry.operations.get("remove_example"), Operation)

    assert len(ds.operations) == 1
    ds.apply_("remove_example")
    assert len(ds.operations) == 2

    assert ds.operations[1].name == "remove_example"
    assert len(ds.example_store) == 2

    assert len(ds) == 0

    print(ds.example_store._map)
