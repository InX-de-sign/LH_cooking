"""Process names for timers, including recipes saved before labels existed."""
import re


def clean_timer_label(value):
    """Keep a compact label within the phone's existing 40-character limit."""
    label = " ".join(str(value or "").split())
    if len(label) > 40:
        label = label[:41].rsplit(" ", 1)[0] if " " in label[:41] else label[:40]
    return label


_ACTIONS = {
    "boil": "Boiling", "simmer": "Simmering", "stir-fry": "Stir-frying",
    "stir fry": "Stir-frying", "fry": "Frying", "bake": "Baking",
    "roast": "Roasting", "steam": "Steaming", "sear": "Searing",
    "brown": "Browning", "rest": "Resting", "chill": "Chilling",
    "refrigerate": "Chilling", "soak": "Soaking", "toast": "Toasting",
    "cook": "Cooking", "heat": "Heating", "warm": "Warming",
    "toss": "Tossing", "stir": "Stirring", "reduce": "Reducing",
}
_ACTION = re.compile(r"\b(" + "|".join(_ACTIONS) + r")\b", re.I)


def timer_label_for_step(step, recipe_name=""):
    explicit = clean_timer_label(step.get("timer_label"))
    if explicit:
        return explicit
    text = " ".join(str(step.get("text") or "").split())
    # Prefer the action attached to a duration, rather than preliminary actions
    # such as mixing batter before baking or moving meat before resting.
    candidates = []
    matches = list(_ACTION.finditer(text))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        clause = text[match.end():end]
        timed = bool(re.search(r"\b(?:seconds?|minutes?|hours?)\b", clause, re.I))
        candidates.append((timed, -i, match, clause))
    if candidates:
        _, _, match, clause = max(candidates, key=lambda item: item[:2])
        action = _ACTIONS[match.group().lower()]
        food = re.split(r"[,.;]|\b(?:for|until|over|with|in|on|at|then|and)\b", clause, maxsplit=1, flags=re.I)[0]
        food = re.sub(r"^(?:the|a|an)\s+", "", food.strip(), flags=re.I)
        if not food or re.fullmatch(r"(?:it|them|gently|briefly|occasionally|often)", food, re.I):
            food = recipe_name
        label = f"{action} {food}".strip()
    else:
        label = text.split(".", 1)[0] or f"Cooking {recipe_name}".strip()
    return clean_timer_label(label)
