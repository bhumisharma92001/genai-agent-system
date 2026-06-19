from utils.logger import logger


class CommandHandler:

    def __init__(self, pipeline, memory_manager, orchestrator):
        self.pipeline = pipeline
        self.memory_manager = memory_manager
        self.orchestrator = orchestrator

    async def handle_ingest(self, query: str, user_id: str, session_id: str) -> None:
        target_path = query[7:].strip()
        if not target_path:
            print("\n[System]: Usage: /ingest <path>")
            return
        try:
            await self.pipeline.index_document(file_path=target_path, user_id=user_id, session_id=session_id)
            print(f"\n[System]: Document '{target_path}' successfully indexed.")
        except Exception as e:
            print(f"\n[Error]: Failed to index document: {e}")

    def handle_list(self, user_id: str, session_id: str) -> None:
        docs = self.pipeline.list_documents(user_id=user_id, session_id=session_id)
        if docs:
            print("\n[System]: Indexed documents:")
            for doc in docs:
                print(f"  - {doc}")
        else:
            print("\n[System]: No documents indexed yet.")

    def handle_delete(self, query: str, user_id: str, session_id: str) -> None:
        source = query[7:].strip()
        if not source:
            print("\n[System]: Usage: /delete <full_file_path>")
            return
        try:
            self.pipeline.delete_document(source=source, user_id=user_id, session_id=session_id)
            print(f"\n[System]: Document '{source}' deleted successfully.")
        except Exception as e:
            print(f"\n[Error]: Failed to delete: {e}")

    def handle_summary(self, user_id: str, session_id: str) -> None:
        summaries = self.memory_manager.get_summaries(user_id=user_id, session_id=session_id)
        print(f"\nSummary:\n{summaries[0][0]}" if summaries else "\nNo session summaries yet.")

    async def handle_startup_ingestion(self, user_id: str, session_id: str) -> None:
        existing_docs = self.pipeline.list_documents(user_id=user_id, session_id=session_id)
        if not existing_docs:
            print("\n[System]: No documents indexed yet. Please upload a document to begin.")
            file_path = input("Enter the path of the document to index: ").strip()
            if file_path:
                try:
                    await self.pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)
                    print(f"\n[System]: Document '{file_path}' successfully indexed.")
                except Exception as e:
                    print(f"\n[Error]: Failed to index document: {e}")
        else:
            print("\n[System]: Documents from your previous session:")
            for doc in existing_docs:
                print(f"  - {doc}")
            file_path = input("\nAdd another document? (press Enter to skip): ").strip()
            if file_path:
                try:
                    await self.pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)
                    print(f"\n[System]: Document '{file_path}' successfully indexed.")
                except Exception as e:
                    print(f"\n[Error]: Failed to index document: {e}")