def test_package_exposes_version():
    import idea_factory

    assert idea_factory.__version__ == "0.1.0"
