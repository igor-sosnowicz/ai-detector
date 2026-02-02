"""Module with LLM-based techniques for LLM detection. Ironic..."""

from pydantic import BaseModel

from src.data_models import Language
from src.nlp.sentence_splitter import NLTKSentenceSplitter
from src.utils import LLM


async def calculate_frequency_of_rule_of_three(
    text: str, language: Language = "english"
) -> float:
    """
    Calculate sentence to the rule of three examples to sentences ratio in a text.

    Args:
        text (str): The text to be assessed.
        language (Language, optional): Language of the text. Defaults to "english".

    Returns:
        float: The ratio calculated by dividing the number of the rule of three
            occurrences by dividing the number of sentences in the text.
    """
    sentence_splitter = NLTKSentenceSplitter()
    system_prompt = """
        You are a helpful text editor completing the tasks.

        The rule of three is a writing principle which suggests that a trio of entities
        such as events or characters is more satisfying, effective, or humorous than
        other numbers, hence also more memorable, because it combines both brevity and
        rhythm with the smallest amount of information needed to create a pattern.

        Count the number of occurrences in the user's text.
        Fill the form in the JSON format.
    """
    llm = LLM(system_prompt=system_prompt)

    class CountingForm(BaseModel):
        occurrences_of_rule_of_three: int

    response: CountingForm = await llm.get_structured_response(
        prompt=text, structured_response_model=CountingForm
    )
    sentences = sentence_splitter.split_into_sentences(text=text, language=language)
    return response.occurrences_of_rule_of_three / len(sentences)
