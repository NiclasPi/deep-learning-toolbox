from dltoolbox.dataset.metadata import DatasetMetadata


def test_metadata_serialization() -> None:
    meta = DatasetMetadata(
        name="Test Dataset",
        split="train",
        origin_path="/path/to/dataset/origin/",
        sample_ids=["01", "02", "03", "04"],
        sample_meta={
            "01": "some metadata string",
            "02": 123.456,
            "03": ["col1", "col2", "col3"],
            "04": {"field": "value"},
            # careful: tuple will turn into list
        }
    )

    meta_encoded = meta.to_json_bytes()
    meta_decoded = DatasetMetadata.from_json_bytes(meta_encoded)
    assert meta == meta_decoded
