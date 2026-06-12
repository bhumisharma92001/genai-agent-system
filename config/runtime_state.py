import sqlite3
import uuid

from exceptions.custom_errors import InvalidInputError, SessionError
from utils.logger import logger


class RuntimeState:

    def __init__(self):
        self.user_id = None
        self.session_id = None

    def initialize(self, memory_manager=None):
        self.user_id = input("Enter user name: ").strip()
        if not self.user_id:
            raise InvalidInputError("User ID cannot be empty")

        if memory_manager is not None:
            self.session_id = self._select_session(memory_manager)
        else:
            self.session_id = str(uuid.uuid4())

        logger.info(f"User initialized: {self.user_id}")
        logger.info(f"Session: {self.session_id}")

        print("\n======================")
        print(f"Welcome {self.user_id}")
        print(f"Session ID: {self.session_id}")
        print("======================\n")

    def _select_session(self, memory_manager) -> str:
        """
        Fetch all sessions for this user and let them pick one.
        If no sessions exist → create new automatically.
        Infrastructure failures → raise SessionError (not silently swallowed).
        """
        try:
            sessions = memory_manager.get_all_sessions(self.user_id)
        except (sqlite3.OperationalError, sqlite3.DatabaseError, OSError, PermissionError) as e:
            raise SessionError(
                f"Cannot access memory database: {e}. "
                "Check DB file path, permissions, and disk space."
            ) from e
        except Exception as e:
            logger.warning(f"Could not fetch sessions: {e}. Starting fresh.")
            return str(uuid.uuid4())

        if not sessions:
            logger.info("No previous sessions found. Creating new session.")
            return str(uuid.uuid4())

        # Show session menu
        print("\n======================")
        print(f"Welcome back, {self.user_id}!")
        print("Your sessions:\n")
        for i, (session_id, created_at, updated_at) in enumerate(sessions, start=1):
            # Mark most recent session (first in list since ordered by updated_at DESC)
            marker = " ← last used" if i == 1 else ""
            print(f"  {i}. Session ID : {session_id}")
            print(f"     Started    : {created_at}")
            print(f"     Last used  : {updated_at}{marker}\n")

        print(f"  n. Start new session")
        print("======================")

        choice = input("Choose [1-{} or n]: ".format(len(sessions))).strip().lower()

        if choice == "n":
            new_id = str(uuid.uuid4())
            logger.info(f"User chose new session: {new_id}")
            return new_id

        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(sessions):
                chosen = sessions[index][0]
                logger.info(f"User resumed session: {chosen}")
                return chosen

        logger.warning(f"Invalid choice '{choice}'. Defaulting to most recent session.")
        return sessions[0][0]