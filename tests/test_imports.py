def test_import_package():
    import image_recommender

    assert hasattr(image_recommender, "__version__")
