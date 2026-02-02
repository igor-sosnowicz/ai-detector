"""Module with a prompt manager interface and implementation."""

import sqlite3
import types
import uuid
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Final, Self, override

from src.configuration import config


class PromptManager(AbstractContextManager, ABC):
    """An interface for a prompt manager."""

    @abstractmethod
    def get_prompts(
        self, uuids: str | list[str] | None, limit: int = 0
    ) -> dict[str, str]:
        """
        Get prompts matching the UUIDs.

        Args:
            uuids (str | list[str] | None): Either a single UUID or multiple UUIDs to
                get prompts with provided UUIDs. None to get all prompts.
            limit (int, optional): The maximum number of prompts to retrieve.
                Useful when `uuids` is None. Defaults to 0.

        Returns:
            dict[str, str]: Mapping of UUIDs to corresponding prompts.
        """

    @abstractmethod
    def add_prompt(self, prompt: str) -> str:
        """
        Add a prompt to the database.

        Args:
            prompt (str): Prompt to be added.

        Returns:
            str: UUID of the prompt that can be used to retrieve the prompted
                from the database later.
        """

    @abstractmethod
    def prompt_exists(self, prompt: str) -> bool:
        """
        Check if a prompt already exists in the database.

        Args:
            prompt (str): Prompt to be verified.

        Returns:
            bool: Whether the prompt exists in the database (True) or not (False).
        """


class SQLitePromptManger(PromptManager):
    """SQLite3-based implementation of a prompt manager."""

    table: Final = "prompts"

    def __init__(self, database_file: Path = config.sqlite_database) -> None:
        """Set the path to a SQLite database file."""
        self._database_file = database_file
        self._connection = None

    def __enter__(self) -> Self:
        """Open the SQLite connection and create the necessary table if missing."""
        self._connection = sqlite3.connect(
            self._database_file,
            check_same_thread=False,
        )
        self._connection.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                uuid TEXT PRIMARY KEY,
                prompt TEXT
            );
        """)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> bool | None:
        """Exit the context manager closing any database connections."""
        if self._connection is not None:
            self._connection.close()

    @override
    def add_prompt(self, prompt: str) -> str:
        if self._connection is None:
            raise ConnectionError(
                "Use the context manager pattern (`with` keyword) to connect to the "
                f"database before calling `{self.add_prompt.__name__}(...)`."
            )

        _uuid = str(uuid.uuid4())
        sql_query = self._construct_sql_insert_query()
        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(sql_query, [_uuid, prompt])
            self._connection.commit()
            return _uuid
        finally:
            if cursor is not None:
                cursor.close()

    def _construct_sql_insert_query(self) -> str:
        return f"INSERT INTO {self.table} VALUES (?, ?);"  # noqa: S608, SQL injection is impossible here.

    @override
    def get_prompts(
        self, uuids: str | list[str] | None, limit: int = 0
    ) -> dict[str, str]:
        if self._connection is None:
            raise ConnectionError(
                "Use the context manager pattern (`with` keyword) to connect to the "
                f"database before calling `{self.get_prompts.__name__}(...)`."
            )

        sql_query = self._construct_sql_select_query(uuids, limit)

        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(sql_query)
            return {str(row[0]): str(row[1]) for row in cursor.fetchall()}

        finally:
            if cursor is not None:
                cursor.close()

    def _construct_sql_select_query(
        self, uuids: str | list[str] | None, limit: int = 0
    ) -> str:
        sql_query = f"SELECT uuid, prompt FROM {self.table}"  # noqa: S608, query injection is impossible with a constant string.

        if isinstance(uuids, str):
            binding_symbols = "?," * len(uuids)
            binding_symbols = binding_symbols[:-1]
            sql_query = f"{sql_query} WHERE uuid IN ({binding_symbols})"
        elif isinstance(uuids, str):
            sql_query = f"{sql_query} WHERE uuid = ?"
        elif uuids is None:
            pass

        return f"{sql_query} LIMIT {limit};" if limit > 0 else f"{sql_query};"

    @override
    def prompt_exists(self, prompt: str) -> bool:
        if self._connection is None:
            raise ConnectionError(
                "Use the context manager pattern (`with` keyword) to connect to the "
                f"database before calling `{self.prompt_exists.__name__}(...)`."
            )

        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(
                f"SELECT 1 FROM {self.table} WHERE prompt = ? LIMIT 1;",  # noqa: S608
                [prompt],  # Constant table name does not pose a name of SQL injection.
            )
            return cursor.fetchone() is not None
        finally:
            if cursor is not None:
                cursor.close()
