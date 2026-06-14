import os
from dotenv import load_dotenv

from config.runtime_state import RuntimeState
from config.system_builder import build_system
from config.command_handler import CommandHandler
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

    handler = CommandHandler(pipeline, memory_manager, orchestrator)
    handler.handle_startup_ingestion(user_id, session_id)

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
            handler.handle_ingest(query, user_id, session_id)
            continue

        if query.startswith("/list"):
            handler.handle_list(user_id, session_id)
            continue

        if query.startswith("/delete"):
            handler.handle_delete(query, user_id, session_id)
            continue

        if query.startswith("/summary"):
            handler.handle_summary(user_id, session_id)
            continue

        try:
            answer = orchestrator.get_response(query=query, user_id=user_id, session_id=session_id)
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            print("\nAssistant: I encountered an issue processing your request. Please try again.")
            continue

        if not answer.strip():
            print("\nAssistant: I'm sorry, I encountered an issue. Please try again.")
            continue

        print(f"\nAssistant: {answer}")

        if memory_manager.should_store(query, answer):
            memory_manager.save_interaction(user_id=user_id, session_id=session_id, query=query, answer=answer)

        interaction_count += 1
        if interaction_count % 3 == 0:
            memory_manager.summarize_in_background(user_id=user_id, session_id=session_id, blocking=False)


if __name__ == "__main__":
    main()