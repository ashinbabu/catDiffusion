from catdiffusion.dataset import CAT_BREEDS
def test_explicit_cat_metadata():
    assert len(CAT_BREEDS)==12
    assert "Labrador_Retriever" not in CAT_BREEDS
