import catalogue


class registry:
    operations = catalogue.create("recon", "operations", entry_points=True)
    operation_factories = catalogue.create("recon", "operation_factories", entry_points=True)
