from recon.operations import Operation, operation, registry


def test_operation_init():
    @operation("test_operation")
    def operation_test(example):
        return example

    assert "test_operation" in registry.operations
    assert isinstance(registry.operations.get("test_operation"), Operation)
