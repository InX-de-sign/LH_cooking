"""Serving-size scaling and recipe-link import (no network: DNS and HTTP are mocked)."""
import asyncio
import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chatbot"))
from scaling import factor_for, scale_ingredient, scale_step
from recipe_import import RecipeImportError, fetch_recipe_page, recipe_text_from_html


class ScalingTests(unittest.TestCase):
    def test_ingredients_scale_their_leading_quantity_in_kitchen_fractions(self):
        cases = {
            "1 tablespoon light soy sauce": ("2 tablespoons light soy sauce", "1/2 tablespoon light soy sauce"),
            "half a teaspoon chilli flakes": ("1 teaspoon chilli flakes", "1/4 teaspoon chilli flakes"),
            "quarter teaspoon salt": ("1/2 teaspoon salt", "1/8 teaspoon salt"),
            "200 grams spaghetti": ("400 grams spaghetti", "100 grams spaghetti"),
            "three quarters cup flour": ("1 1/2 cups flour", "1/3 cup flour"),
            "2 garlic cloves, minced": ("4 garlic cloves, minced", "1 garlic cloves, minced"),
            "salt to taste": ("salt to taste", "salt to taste"),
            "1 400 gram tin tomatoes": ("2 400 gram tin tomatoes", "1/2 400 gram tin tomatoes"),
        }
        for line, (double, half) in cases.items():
            with self.subTest(line=line):
                self.assertEqual(scale_ingredient(line, Fraction(2)), double)
                self.assertEqual(scale_ingredient(line, Fraction(1, 2)), half)
        self.assertEqual(scale_ingredient("150 grams flour", Fraction(3, 2)), "225 grams flour", "metric rounds to 5")

    def test_steps_scale_measured_amounts_but_never_times_or_temperatures(self):
        step = "Add 2 tablespoons soy sauce and 150 g flour, then cook for 3 minutes at 200 C."
        self.assertEqual(scale_step(step, Fraction(2)),
                         "Add 4 tablespoons soy sauce and 300 g flour, then cook for 3 minutes at 200 C.")
        self.assertEqual(scale_step("Beat 1 egg with half a teaspoon salt.", Fraction(2)),
                         "Beat 1 egg with 1 teaspoon salt.")
        untouched = "Toss for one to two minutes with about half the water."
        self.assertEqual(scale_step(untouched, Fraction(2)), untouched)

    def test_factor(self):
        self.assertEqual(factor_for(2, 4), Fraction(2))
        for base, wanted in ((None, 4), (2, 2), (0, 3), ("x", 2), (2, None)):
            self.assertIsNone(factor_for(base, wanted))


class RecipeLinkTests(unittest.TestCase):
    PAGE = """<html><head><script type="application/ld+json">
    {"@context": "https://schema.org", "@graph": [{"@type": "WebPage"},
      {"@type": ["Recipe"], "name": "Garlic Noodles &amp; Butter", "recipeYield": ["2", "2 servings"],
       "recipeIngredient": ["200 g egg noodles", "4 cloves garlic"],
       "recipeInstructions": [{"@type": "HowToSection", "name": "Noodles",
           "itemListElement": [{"@type": "HowToStep", "text": "Boil the noodles for 3 minutes."}]},
         {"@type": "HowToStep", "text": "Fry the <b>garlic</b> in butter."}]}]}
    </script></head><body><nav>Menu</nav><p>Life story...</p></body></html>"""

    def test_schema_org_recipe_is_preferred_over_page_text(self):
        text = recipe_text_from_html(self.PAGE)
        self.assertTrue(text.startswith("Garlic Noodles & Butter\nServes: 2\nIngredients:"))
        self.assertIn("- 200 g egg noodles", text)
        self.assertIn("1. Noodles:", text)
        self.assertIn("2. Boil the noodles for 3 minutes.", text)
        self.assertIn("Fry the  garlic  in butter.", text)
        self.assertNotIn("Life story", text)

    def test_page_without_structured_data_falls_back_to_visible_text(self):
        text = recipe_text_from_html("<script>var x=1</script><h1>Toast</h1><p>Toast the bread.</p><style>p{}</style>")
        self.assertEqual(text, "Toast\nToast the bread.")

    def run_fetch(self, url, addresses=("93.184.216.34",), handler=None):
        import httpx

        async def getaddrinfo(host, port, type=0):
            return [(2, 1, 6, "", (address, 0)) for address in addresses]

        transport = httpx.MockTransport(handler or (lambda request: httpx.Response(
            200, headers={"content-type": "text/html; charset=utf-8"}, text=self.PAGE)))
        real_client = httpx.AsyncClient

        async def go():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo", getaddrinfo), \
                    patch("httpx.AsyncClient", lambda **kw: real_client(transport=transport, **kw)):
                return await fetch_recipe_page(url)
        return asyncio.run(go())

    def test_fetch_reads_a_public_recipe_page(self):
        self.assertIn("Garlic Noodles", self.run_fetch("https://recipes.example/noodles"))

    def test_fetch_refuses_private_addresses_bad_schemes_and_ports(self):
        for address in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "::1", "192.168.1.2"):
            with self.subTest(address=address), self.assertRaisesRegex(RecipeImportError, "private"):
                self.run_fetch("https://sneaky.example/", addresses=(address,))
        for url in ("file:///etc/passwd", "ftp://x.example/r", "https://user:pw@x.example/", "https://x.example:8000/"):
            with self.subTest(url=url), self.assertRaises(RecipeImportError):
                self.run_fetch(url)

    def test_redirects_are_checked_again(self):
        import httpx
        seen = []

        def handler(request):
            seen.append(str(request.url))
            return httpx.Response(302, headers={"location": "http://internal.example/admin"})

        async def getaddrinfo(host, port, type=0):
            return [(2, 1, 6, "", ("10.1.2.3" if host == "internal.example" else "93.184.216.34", 0))]

        transport = httpx.MockTransport(handler)
        real_client = httpx.AsyncClient

        async def go():
            loop = asyncio.get_running_loop()
            with patch.object(loop, "getaddrinfo", getaddrinfo), \
                    patch("httpx.AsyncClient", lambda **kw: real_client(transport=transport, **kw)):
                return await fetch_recipe_page("https://recipes.example/r")
        with self.assertRaisesRegex(RecipeImportError, "private"):
            asyncio.run(go())
        self.assertEqual(seen, ["https://recipes.example/r"], "the private redirect target is never requested")

    def test_non_html_and_errors_are_explained(self):
        import httpx
        with self.assertRaisesRegex(RecipeImportError, "isn't a web page"):
            self.run_fetch("https://x.example/a.pdf", handler=lambda r: httpx.Response(
                200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
        with self.assertRaisesRegex(RecipeImportError, "HTTP 404"):
            self.run_fetch("https://x.example/gone", handler=lambda r: httpx.Response(404, text="no"))


if __name__ == "__main__":
    unittest.main()
