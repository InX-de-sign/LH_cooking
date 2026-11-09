# scaling.py - Rescale a recipe's quantities to a different number of servings
"""
"Make it for 4" should change the numbers the cook reads, not just the
model's next reply. Two rules keep this safe:

    ingredient lines   the leading quantity is scaled, whatever follows it
                       ("2 garlic cloves" -> "4 garlic cloves")
    step text          only a quantity directly followed by a measuring unit
                       is scaled ("add 2 tablespoons soy sauce"); times,
                       temperatures and counts like "1 egg" are left alone

Fractions come out the way people write them in a kitchen (1/2, 1 1/3);
metric weights and volumes above 20 round to the nearest 5.
"""

import re
from fractions import Fraction

WORD_QUANTITIES = {
    "three quarters": Fraction(3, 4), "two thirds": Fraction(2, 3),
    "one third": Fraction(1, 3), "a third": Fraction(1, 3),
    "half an": Fraction(1, 2), "half a": Fraction(1, 2), "half": Fraction(1, 2),
    "a quarter": Fraction(1, 4), "quarter": Fraction(1, 4),
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
UNICODE_FRACTIONS = {"½": Fraction(1, 2), "¼": Fraction(1, 4), "¾": Fraction(3, 4),
                     "⅓": Fraction(1, 3), "⅔": Fraction(2, 3)}

_QTY = (r"\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|[½¼¾⅓⅔]|"
        + "|".join(sorted((re.escape(w) for w in WORD_QUANTITIES), key=len, reverse=True)))
_UNIT = (r"grams?|g|kg|kilograms?|millilit(?:re|er)s?|ml|lit(?:re|er)s?|l|"
         r"tablespoons?|tbsp|teaspoons?|tsp|cups?")
METRIC = re.compile(r"^(grams?|g|kg|kilograms?|millilit(re|er)s?|ml)$", re.I)

LEADING = re.compile(rf"^(?P<q>{_QTY})\b(?P<rest>.*)$", re.I | re.S)
INLINE = re.compile(rf"\b(?P<q>{_QTY})(?P<gap>\s+(?:of\s+)?(?:an?\s+)?|\s*)(?P<u>{_UNIT})\b", re.I)
_NICE = (Fraction(0), Fraction(1, 8), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2),
         Fraction(2, 3), Fraction(3, 4), Fraction(1))
_COUNTABLE_UNITS = ("tablespoon", "teaspoon", "cup", "gram", "kilogram", "millilitre",
                    "milliliter", "litre", "liter", "clove", "pinch", "handful", "slice")


def _agree(unit_and_rest: str, value: Fraction) -> str:
    """'2 tablespoon' -> '2 tablespoons'; '1/2 tablespoons' -> '1/2 tablespoon'."""
    match = re.match(r"^(\s*)([A-Za-z]+)(.*)$", unit_and_rest, re.S)
    if not match:
        return unit_and_rest
    space, word, rest = match.groups()
    lower = word.lower()
    singular = lower[:-2] if lower.endswith("ches") else lower[:-1] if lower.endswith("s") else lower
    if singular not in _COUNTABLE_UNITS:
        return unit_and_rest
    plural = singular + ("es" if singular.endswith("ch") else "s")
    chosen = plural if value > 1 else singular
    if word[0].isupper():
        chosen = chosen.capitalize()
    return f"{space}{chosen}{rest}"


def parse_quantity(text: str):
    text = " ".join(text.lower().split())
    if text in WORD_QUANTITIES:
        return Fraction(WORD_QUANTITIES[text])
    if text in UNICODE_FRACTIONS:
        return UNICODE_FRACTIONS[text]
    parts = text.split()
    try:
        return sum((Fraction(p) for p in parts), Fraction(0))
    except (ValueError, ZeroDivisionError):
        return None


def format_quantity(value: Fraction, metric: bool = False) -> str:
    if metric and value > 20:
        return str(int(round(float(value) / 5) * 5))
    whole = int(value)
    rest = value - whole
    nice = min(_NICE, key=lambda f: abs(f - rest))
    if nice == 1:
        whole, nice = whole + 1, Fraction(0)
    if nice == 0:
        return str(whole) if whole else "1/8"
    frac = f"{nice.numerator}/{nice.denominator}"
    return f"{whole} {frac}" if whole else frac


def scale_ingredient(line: str, factor: Fraction) -> str:
    match = LEADING.match(line or "")
    if not match or factor == 1:
        return line
    quantity = parse_quantity(match.group("q"))
    if quantity is None:
        return line
    rest = match.group("rest")
    unit = rest.split()[0] if rest.split() else ""
    # "half a teaspoon" -> "1 teaspoon", not "1 a teaspoon".
    rest = re.sub(r"^\s+(an?\s+)", " ", rest) if match.group("q").lower().startswith("half") else rest
    scaled = quantity * factor
    return format_quantity(scaled, bool(METRIC.match(unit))) + _agree(rest, scaled)


def scale_step(text: str, factor: Fraction) -> str:
    if factor == 1:
        return text

    def replace(match):
        quantity = parse_quantity(match.group("q"))
        if quantity is None:
            return match.group(0)
        gap = match.group("gap")
        if match.group("q").lower().startswith("half"):
            gap = " "
        scaled = quantity * factor
        unit = match.group("u")
        return (format_quantity(scaled, bool(METRIC.match(unit)))
                + (_agree(gap + unit, scaled) if gap.strip() == "" and gap else gap + unit))

    return INLINE.sub(replace, text or "")


def factor_for(base_serves, servings):
    """Fraction(servings, base_serves), or None when scaling does not apply."""
    try:
        base, wanted = int(base_serves or 0), int(servings or 0)
    except (TypeError, ValueError):
        return None
    if base <= 0 or wanted <= 0 or base == wanted:
        return None
    return Fraction(wanted, base)
