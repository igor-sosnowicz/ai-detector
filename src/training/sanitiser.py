"""Module with a text sanitiser."""

import re


def sanitise(text: str) -> str:
    """
    Remove formatting artifacts and normalise text for fair stylometric analysis.

    This removes markdown, HTML, and other formatting that could bias the model
    toward detecting format rather than writing style.

    Args:
        text (str): Raw text potentially containing formatting.

    Returns:
        str: Cleaned text with only content and natural punctuation.
    """
    # Remove HTML tags.
    text = re.sub(r"<[^>]+>", "", text)

    # Remove markdown headers (# ## ###).
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Remove markdown emphasis/bold (**text**, *text*, __text__, _text_).
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)

    # Remove markdown links [text](url) -> text.
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

    # Remove markdown images ![alt](url).
    text = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", "", text)

    # Remove markdown code blocks ```code```.
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)

    # Remove inline code `code`.
    text = re.sub(r"`([^`]+)`", r"\1", text)

    # Remove markdown list bullets (* - +).
    text = re.sub(r"^\s*[\*\-\+]\s+", "", text, flags=re.MULTILINE)

    # Remove numbered lists (1. 2. etc).
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove markdown blockquotes (>).
    text = re.sub(r"^\s*>\s+", "", text, flags=re.MULTILINE)

    # Remove horizontal rules (---, ***, ___).
    text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove excessive whitespace (multiple spaces, tabs).
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive newlines (3+ → 2).
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove leading/trailing whitespace on each line.
    text = "\n".join(line.strip() for line in text.split("\n"))

    # Remove leading/trailing whitespace from entire text.
    return text.strip()
