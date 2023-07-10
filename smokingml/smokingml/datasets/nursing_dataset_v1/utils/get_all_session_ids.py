from pathlib import Path

def get_all_session_ids(dir: Path) -> list[int]:
    # Get list of all session ids in dataset

    session_ids = []
    for session_id in dir.iterdir():
        session_ids.append(int(session_id.name))

    return session_ids