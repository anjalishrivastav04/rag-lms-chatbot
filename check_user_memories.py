"""
check_user_memories.py
------------------------
Dumps all mem0 facts currently stored for a given user_id, so you can spot
stale/incorrect facts (e.g. a leftover "favorite color: red" from earlier
testing) before deciding whether to delete them.

Usage (run from project root):
    python check_user_memories.py <user_id>
    python check_user_memories.py <user_id> --delete <memory_id>   # delete one fact
    python check_user_memories.py <user_id> --delete-all           # wipe ALL facts for this user
"""

import argparse
from services.memory_service import memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("user_id")
    parser.add_argument("--delete", help="Delete a single memory by its id")
    parser.add_argument("--delete-all", action="store_true", help="Delete ALL memories for this user")
    args = parser.parse_args()

    user_id = str(args.user_id)

    if args.delete:
        memory.delete(memory_id=args.delete)
        print(f"✅ Deleted memory {args.delete}")
        return

    if args.delete_all:
        confirm = input(f"Type 'yes' to permanently delete ALL memories for user_id={user_id}: ")
        if confirm.strip().lower() == "yes":
            memory.delete_all(user_id=user_id)
            print(f"✅ Deleted all memories for user_id={user_id}")
        else:
            print("Cancelled.")
        return

    result = memory.get_all(user_id=user_id)
    facts = result.get("results", []) if isinstance(result, dict) else result
    if not facts:
        print(f"No memories found for user_id={user_id}")
        return

    print(f"=== {len(facts)} memories for user_id={user_id} ===")
    for f in facts:
        print(f"  id={f.get('id')}  memory={f.get('memory')!r}")


if __name__ == "__main__":
    main()