"""Module with project-wide utilities."""

import asyncio
import hashlib
import json
from typing import TypeVar

from cerebras.cloud.sdk import Cerebras
from diskcache import Cache
from pydantic import BaseModel

from src.configuration import config
from src.data_models import ChatMessage

T = TypeVar("T", bound=BaseModel)


class ChatHistory:
    """History of a chat with an LLM convertible to OpenAI compatible format."""

    def __init__(self, system_prompt: str) -> None:
        """
        Initialise an new chat with an empty cache.

        Args:
            system_prompt (str): System prompt for an LLM to be used in the chat.
        """
        self._history: list[ChatMessage] = [
            ChatMessage(message=system_prompt, role="system")
        ]
        self._state_changed = False
        self._cache = []
        self._hash = ""

    def add_message(self, message: ChatMessage) -> None:
        """
        Add a new message to chat.

        Args:
            message (ChatMessage): Message to be added.
        """
        self._history.append(message)
        self._state_changed = True

    def to_raw(self) -> list[dict[str, str]]:
        """
        Convert current chat history to an OpenAI-compatible format.

        Returns:
            list[dict[str, str]]: OpenAI-compatible chat history.
        """
        if not self._state_changed:
            return self._cache

        self._cache = [message.to_dict() for message in self._history]
        self._state_changed = False
        return self._cache


class LLM:
    """Wrapper for handling chats with LLMs using a remote inference provider."""

    cache = Cache(config.llm_cache_directory, size_limit=2**27)  # 128 MB

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "gpt-oss-120b",
    ) -> None:
        """
        Initialise a chat with a remote LLM.

        Args:
            system_prompt (str, optional): System prompt for the LLM.
                Defaults to "You are a helpful assistant.".
            model (str, optional): Name of the model to be used in
                the entire chat history. Defaults to "gpt-oss-120b".

        Raises:
            ValueError: Raised if LLM inference provider API key is missing.
        """
        config.llm_cache_directory.mkdir(exist_ok=True, parents=True)
        self.model = model
        self._chat_history = ChatHistory(system_prompt)
        api_key = config.cerebras_api_key
        if not api_key:
            raise ValueError(
                "Cerebras API key is not set. "
                "Provide config.cerebras_api_key or set CEREBRAS_API_KEY."
            )
        self._cerebras_client = Cerebras(api_key=api_key)

    def _to_cache_key(self, *, structured_response_expected: bool) -> str:
        data = (
            f"{structured_response_expected}|{self.model}|{self._chat_history.to_raw()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def _build_json_schema_response_format(self, model: type[BaseModel]) -> dict:
        """Build a JSON schema response format for Cerebras Inference."""
        schema = model.model_json_schema()

        # Extract required fields.
        required_fields = schema.get("required", [])

        # Build properties with descriptions.
        properties = {}
        for field_name, field_info in schema.get("properties", {}).items():
            properties[field_name] = {
                "type": field_info.get("type", "string"),
                "description": field_info.get("description", ""),
            }

        return {
            "type": "json_schema",
            "json_schema": {
                "name": model.__name__,
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required_fields,
                    "additionalProperties": False,
                },
            },
        }

    async def _handle_structured_output_query(
        self, structured_response_model: type[T]
    ) -> T:
        """Get structured output from LLM using JSON schema format."""
        key = self._to_cache_key(structured_response_expected=True)
        if key in self.cache:
            response_text = str(self.cache[key])
        else:
            response_format = self._build_json_schema_response_format(
                structured_response_model
            )

            completion = await asyncio.to_thread(
                self._cerebras_client.chat.completions.create,
                model=self.model,
                messages=self._chat_history.to_raw(),
                response_format=response_format,
            )

            response_text = completion.choices[0].message.content
            if response_text is None:
                raise ValueError("The LLM returned an empty response.")

            self.cache[key] = str(response_text)

        # Parse the JSON response
        json_data = json.loads(response_text)
        return structured_response_model.model_validate(json_data)

    async def _handle_unstructured_output_query(self) -> str:
        key = self._to_cache_key(structured_response_expected=False)
        if key in self.cache:
            return str(self.cache[key])

        completion = await asyncio.to_thread(
            self._cerebras_client.chat.completions.create,
            model=self.model,
            messages=self._chat_history.to_raw(),
        )

        response = completion.choices[0].message.content
        if response is None:
            raise ValueError("The response from the LLM is empty.")

        self.cache[key] = str(response)
        return response

    async def get_structured_response(
        self, prompt: str, structured_response_model: type[T]
    ) -> T:
        """
        Get a structured response from an LLM in a chat.

        Args:
            prompt (str): Prompt to be sent to the LLM to explain it
                how the output should be filled.
            structured_response_model (type[T]): Type of the response derived from
                the Pydantic `BaseModel`.

        Returns:
            T: A structured output generated by the LLM.
        """
        self._chat_history.add_message(ChatMessage(message=prompt, role="user"))
        response = await self._handle_structured_output_query(structured_response_model)

        textual_response = response.model_dump_json()
        self._chat_history.add_message(
            ChatMessage(message=textual_response, role="assistant")
        )

        return response

    async def get_response(self, prompt: str) -> str:
        """
        Get an unstructured response in a chat.

        Args:
            prompt (str): Prompt to be sent to an LLM.

        Returns:
            str: Response of the LLM.
        """
        self._chat_history.add_message(
            ChatMessage(message=prompt, role="user"),
        )
        response = await self._handle_unstructured_output_query()
        self._chat_history.add_message(ChatMessage(message=response, role="assistant"))
        return response
