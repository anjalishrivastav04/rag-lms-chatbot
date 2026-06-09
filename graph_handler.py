import os
from neo4j import GraphDatabase
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# --- NEO4J CONNECTION ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# --- GROQ LLM ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

# -------------------------------------------
# EXTRACT ENTITIES & RELATIONSHIPS FROM TEXT
# -------------------------------------------
def extract_graph_data(text, source_filename):
    """Use LLM to extract entities and relationships from text chunk."""
    
    prompt = f"""You are a knowledge graph expert. Extract entities and relationships from the text below.

Return ONLY a JSON array like this (no explanation, no markdown, just raw JSON):
[
  {{"entity1": "Python", "relation": "IS_A", "entity2": "Programming Language"}},
  {{"entity1": "Flask", "relation": "USED_FOR", "entity2": "Web Development"}}
]

Rules:
- Extract maximum 10 most important relationships
- Keep entity names short and clear (2-4 words max)
- Use UPPERCASE_WITH_UNDERSCORES for relations
- If no clear relationships found, return empty array: []

Text:
{text[:1500]}

JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean up response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        import json
        relationships = json.loads(content)
        return relationships
    except Exception as e:
        print(f"⚠️ Entity extraction failed: {e}")
        return []


# -------------------------------------------
# STORE GRAPH DATA IN NEO4J
# -------------------------------------------
def store_in_graph(relationships, source_filename):
    """Store extracted entities and relationships in Neo4j."""
    
    if not relationships:
        return 0
    
    stored_count = 0
    
    try:
        with get_driver().session() as session:
            for rel in relationships:
                entity1 = rel.get("entity1", "").strip()
                relation = rel.get("relation", "").strip().upper().replace(" ", "_")
                entity2 = rel.get("entity2", "").strip()
                
                if not entity1 or not relation or not entity2:
                    continue
                
                # Create nodes and relationship in Neo4j
                query = """
                MERGE (a:Entity {name: $entity1})
                MERGE (b:Entity {name: $entity2})
                MERGE (a)-[r:RELATION {type: $relation, source: $source}]->(b)
                RETURN a, b, r
                """
                
                session.run(query, 
                    entity1=entity1,
                    entity2=entity2,
                    relation=relation,
                    source=source_filename
                )
                stored_count += 1
                
        print(f"✅ Stored {stored_count} relationships in Neo4j graph")
        return stored_count
        
    except Exception as e:
        print(f"❌ Neo4j store error: {e}")
        return 0


# ------------------------------------------- 
# PROCESS CHUNKS AND BUILD GRAPH
# -------------------------------------------
def build_graph_from_chunks(chunks, source_filename):
    """Process document chunks and build knowledge graph."""
    
    # Check if already graphed
    try:
        with get_driver().session() as session:
            result = session.run(
                "MATCH ()-[r:RELATION {source: $source}]->() RETURN count(r) as count",
                source=source_filename
            )
            count = result.single()["count"]
            if count > 0:
                print(f"⚡ Graph already exists for {source_filename} ({count} relationships) — skipping!")
                return count
    except Exception as e:
        print(f"⚠️ Graph check error: {e}")

    print(f"\n🔗 Building knowledge graph for: {source_filename}")
    total_relationships = 0
    
    for i, chunk in enumerate(chunks):
        if i % 3 != 0:
            continue
            
        text = chunk.page_content
        if len(text.strip()) < 100:
            continue
        
        print(f"   🧠 Extracting entities from chunk {i+1}/{len(chunks)}...")
        relationships = extract_graph_data(text, source_filename)
        
        if relationships:
            count = store_in_graph(relationships, source_filename)
            total_relationships += count
    
    print(f"✅ Graph built: {total_relationships} total relationships for {source_filename}")
    return total_relationships
# -------------------------------------------
# QUERY GRAPH FOR RELEVANT CONTEXT
# -------------------------------------------
def graph_retrieve(question, top_k=5):
    """Multi-hop graph traversal to find related entities not directly in embeddings."""
    
    try:
        words = [w.strip('?.,!') for w in question.split() if len(w) > 3]
        if not words:
            return "", []

        found_entities = set()
        graph_context = []

        with get_driver().session() as session:
            
            # --- HOP 1: Find direct matches ---
            for word in words[:5]:
                result = session.run("""
                    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                    WHERE toLower(a.name) CONTAINS toLower($keyword)
                       OR toLower(b.name) CONTAINS toLower($keyword)
                    RETURN a.name as entity1, r.type as relation, b.name as entity2
                    LIMIT $limit
                """, keyword=word, limit=top_k)

                for record in result:
                    e1 = record['entity1']
                    e2 = record['entity2']
                    rel = record['relation']
                    triple = f"{e1} --[{rel}]--> {e2}"
                    if triple not in graph_context:
                        graph_context.append(triple)
                    found_entities.add(e1)
                    found_entities.add(e2)

            # --- HOP 2: Find neighbors of found entities ---
            for entity in list(found_entities)[:10]:
                result = session.run("""
                    MATCH (a:Entity)-[r:RELATION]->(b:Entity)
                    WHERE toLower(a.name) CONTAINS toLower($entity)
                       OR toLower(b.name) CONTAINS toLower($entity)
                    RETURN a.name as entity1, r.type as relation, b.name as entity2
                    LIMIT 3
                """, entity=entity.lower())

                for record in result:
                    e1 = record['entity1']
                    e2 = record['entity2']
                    rel = record['relation']
                    triple = f"{e1} --[{rel}]--> {e2}"
                    if triple not in graph_context:
                        graph_context.append(triple)
                    found_entities.add(e1)
                    found_entities.add(e2)

        # --- BUILD CONTEXT STRING ---
        if graph_context:
            context = "=== KNOWLEDGE GRAPH CONTEXT ===\n"
            context += "\n".join(graph_context[:15])
            context += "\n================================\n"
            print(f"🔗 Graph retrieved {len(graph_context)} relationships via multi-hop traversal")
            print(f"🔗 Related entities found: {list(found_entities)[:10]}")
            return context, list(found_entities)

        return "", []

    except Exception as e:
        print(f"⚠️ Graph retrieval error: {e}")
        return "", []
# -------------------------------------------
# DELETE GRAPH DATA FOR A FILE
# -------------------------------------------
def delete_graph_for_file(source_filename):
    """Remove all graph data for a specific file."""
    try:
        with get_driver().session() as session:
            query = """
            MATCH ()-[r:RELATION {source: $source}]->()
            DELETE r
            """
            session.run(query, source=source_filename)
            
            # Clean up orphan nodes
            session.run("""
            MATCH (n:Entity) 
            WHERE NOT (n)--() 
            DELETE n
            """)
            
        print(f"🗑️ Graph data deleted for: {source_filename}")
    except Exception as e:
        print(f"⚠️ Graph delete error: {e}")


# -------------------------------------------
# TEST CONNECTION
# -------------------------------------------
def test_connection():
    try:
        get_driver().verify_connectivity()
        print("✅ Neo4j Graph DB connected!")
        return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()