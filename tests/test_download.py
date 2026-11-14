import pytest

from download import file_name, page_url


@pytest.mark.parametrize("title, url", [
    # apostrophes are percent-encoded, as in the sitemap ...
    ("0x33's_Aviators", "https://terraria.wiki.gg/wiki/0x33%27s_Aviators"),
    # ... while ":", "(", ")" and "/" are kept
    ("Guide:Class_setups", "https://terraria.wiki.gg/wiki/Guide:Class_setups"),
    ("Apple_(disambiguation)", "https://terraria.wiki.gg/wiki/Apple_(disambiguation)"),
    ("Recipes/Hardmode_Anvil", "https://terraria.wiki.gg/wiki/Recipes/Hardmode_Anvil"),
])
def test_page_url(title, url):
    assert page_url(title) == url


def test_file_name_has_no_folders():
    assert file_name("Recipes/Hardmode_Anvil") == "Recipes%2FHardmode_Anvil"
