import os
from dotenv import load_dotenv

from config.runtime_state import RuntimeState
from config.system_builder import build_system
from utils.logger import logger


def main():
    try:
        run_repl_loop(*initialize_system())
    except SystemExit:
        raise
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise SystemExit(1)


def initialize_system():
    load_dotenv()
    pipeline, orchestrator, memory_manager = build_system()
    runtime = RuntimeState()
    runtime.initialize(memory_manager=memory_manager)
    memory_manager.register_session(user_id=runtime.user_id, session_id=runtime.session_id)
    return pipeline, orchestrator, memory_manager, runtime


def run_repl_loop(pipeline, orchestrator, memory_manager, runtime):
    user_id = runtime.user_id
    session_id = runtime.session_id

    file_path = input("Enter the path of the document to index: ").strip()
    if file_path:
        pipeline.index_document(file_path=file_path, user_id=user_id, session_id=session_id)

    interaction_count = 0
    print("\nSystem ready. Type your queries below.")
    print("To index a new document at any time, use: /ingest <file_path>")

    while True:
        query = input("\nYou: ").strip()
        if not query:
            continue

        logger.info(f"User query received: {query}")
        clean_query = query.lower()

        if clean_query in ["exit", "bye"]:
            if memory_manager.get_recent_interactions(user_id=user_id, session_id=session_id, limit=1):
                memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=True)
            print("\nGoodbye!")
            break

        if query.startswith("/ingest"):
            _handle_ingest(query, pipeline, user_id, session_id)
            interaction_count += 1
            if interaction_count % 3 == 0:
                memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=False)
            continue

        if query.startswith("/list"):
            docs = pipeline.list_documents(user_id=user_id, session_id=session_id)
            if docs:
                print("\n[System]: Indexed documents:")
                for doc in docs:
                    print(f"  - {doc}")
            else:
                print("\n[System]: No documents indexed yet.")
            continue

        if query.startswith("/delete"):
            source = query[7:].strip()
            if source:
                try:
                    pipeline.delete_document(source=source, user_id=user_id, session_id=session_id)
                    print(f"\n[System]: Document '{source}' deleted successfully.")
                except Exception as e:
                    print(f"\n[Error]: Failed to delete: {e}")
            else:
                print("\n[System]: Usage: /delete <full_file_path>  (use /list to see exact paths)")
            continue

        if clean_query == "summary":
            summaries = memory_manager.get_summaries(user_id=user_id, session_id=session_id)
            print(f"\nSummary:\n{summaries[0][0]}" if summaries else "\nNo session summaries yet.")
            continue

        answer = orchestrator.get_response(query=query, user_id=user_id, session_id=session_id)

        if not answer.strip():
            print("\nAssistant: I'm sorry, I encountered an issue. Please try again.")
            continue

        print(f"\nAssistant: {answer}")

        if memory_manager.should_store(query, answer):
            memory_manager.save_interaction(user_id=user_id, session_id=session_id, query=query, answer=answer)

        interaction_count += 1
        if interaction_count % 3 == 0:
            memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=False)


def _handle_ingest(query, pipeline, user_id, session_id):
    target_path = query[7:].strip()
    if target_path:
        try:
            pipeline.index_document(file_path=target_path, user_id=user_id, session_id=session_id)
            print(f"\n[System]: Document '{target_path}' successfully indexed.")
        except Exception as e:
            print(f"\n[Error]: Failed to index document: {e}")
    else:
        print("\n[System]: Please provide a valid file path. Usage: /ingest <path>")


if __name__ == "__main__":
    main()