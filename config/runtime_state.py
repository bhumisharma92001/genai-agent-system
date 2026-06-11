import uuid
from utils.logger import logger

class RuntimeState:

    def __init__(self):
        self.user_id = None
        self.session_id = None
    def initialize(self, memory_manager=None):
        self.user_id = input("Enter user name: ").strip()
        if not self.user_id:
            raise ValueError("User ID cannot be empty")

        last_session = None
        if memory_manager is not None:
            try:
                last_session = memory_manager.get_last_session_id(self.user_id)
            except Exception as e:
                logger.warning(f"Could not resume last session: {e}. Starting fresh.")
                last_session = None

        if last_session:
            self.session_id = last_session
            message = "Resumed last session"
        else:
            self.session_id = str(uuid.uuid4())
            message = "Created a new session"

        logger.info(f"User initialized: {self.user_id}")
        logger.info(f"Session created: {self.session_id}")

        print("\n======================")
        print(f"Welcome {self.user_id}")
        print(f"Session ID: {self.session_id}")
        print(f"{message}")
        print("======================\n")