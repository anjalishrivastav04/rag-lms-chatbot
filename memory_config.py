from dotenv import load_dotenv
load_dotenv()

from mem0 import Memory

config = {
    "vector_store": {
        "provider": "pgvector",
        "config": {
            "user": "postgres",
            "password": "Bingos123",
            "host": "localhost",
            "port": "5432",
            "dbname": "mem0_db",
            "embedding_model_dims": 384,
        }
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {
            "url": "bolt://localhost:7687",
            "username": "neo4j",
            "password": "your_actual_neo4j_password",
        }
    },
    "llm": {
        "provider": "groq",
        "config": {
            "model": "llama-3.1-8b-instant",
        }
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "all-MiniLM-L6-v2"
        }
    }
}

memory = Memory.from_config(config)

memory.add("I love hiking and I'm allergic to peanuts", user_id="test_user")
result = memory.search("What are my allergies?", filters={"user_id": "test_user"})
print(result)