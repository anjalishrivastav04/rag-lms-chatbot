import os
import re
import json
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
# Using the larger 70B model specifically for graph extraction (not the
# 8B model used elsewhere) because the comprehensive cybersecurity
# ontology below has 27 entity types and 27 relation types — a small,
# fast model tends to misclassify or default to generic labels when
# given that many options to choose from in one prompt.
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"))


# -------------------------------------------
# CYBERSECURITY ONTOLOGY (Issue #7)
# -------------------------------------------
# Comprehensive fixed entity types, organized by category below (the
# grouping is just for readability in this file — Neo4j sees a flat
# label). Anything the LLM returns outside this set falls back to a
# generic "Entity" label rather than being dropped, so we don't lose a
# real relationship just because of a labeling miss.
VALID_ENTITY_TYPES = {
    # Threat landscape
    "ThreatActor", "Campaign", "Malware", "AttackTechnique", "Tool", "Exploit",
    # Weaknesses & exposure
    "Vulnerability", "Weakness", "Misconfiguration",
    # Indicators of compromise
    "IOC", "IPAddress", "Domain", "URL", "FileHash", "EmailAddress", "C2Infrastructure",
    # Assets & environment
    "Product", "Vendor", "Asset", "Network", "Organization", "Industry", "Person",
    # Defense
    "SecurityControl", "Mitigation", "DetectionRule", "Patch", "SecurityTeam",
    # Governance & compliance
    "Standard", "Regulation", "Policy",
    # Context
    "Country", "Incident",
}

# Comprehensive fixed relationship vocabulary. Generic relations
# (RELATED_TO, CONNECTED_WITH, ASSOCIATED_WITH, etc.) are intentionally
# NOT in this list — triples using them are rejected rather than
# stored, per the issue's technical notes.
VALID_RELATIONS = {
    "AFFECTS", "EXPLOITS", "TARGETS", "USES", "MITIGATES",
    "MITIGATED_BY", "DETECTS", "DETECTED_BY", "PATCHED_BY",
    "ATTRIBUTED_TO", "EXECUTES", "INDICATES", "DEVELOPED_BY",
    "BELONGS_TO", "COMMUNICATES_WITH", "DOWNLOADS", "DEPLOYS",
    "COMPROMISES", "LEADS_TO", "REQUIRES", "LOCATED_IN", "OPERATES_IN",
    "COMPLIES_WITH", "VIOLATES", "RESPONDS_TO", "IDENTIFIED_BY", "SIMILAR_TO",
    "STEALS", "CAPTURES", "EXFILTRATES", "BYPASSES", "EVADES",
    "ALTERS", "CORRUPTS", "ENCRYPTS", "SPREADS_VIA", "IMPERSONATES",
}

# Small, extendable alias map for entity normalization. Add to this
# over time as you encounter more synonyms in real documents.
ENTITY_ALIASES = {
    "apache": "Apache HTTP Server",
    "apache server": "Apache HTTP Server",
    "httpd": "Apache HTTP Server",
    "cozy bear": "APT29",
    "the dukes": "APT29",
}


def normalize_entity_name(name: str) -> str:
    """
    Normalizes an entity name for deduplication:
    - trims/collapses whitespace
    - standardizes CVE IDs to the canonical CVE-YYYY-NNNN format
    - applies known aliases (case-insensitive) so synonyms merge into
      one node instead of creating duplicates
    """
    if not name:
        return name
    cleaned = re.sub(r"\s+", " ", name.strip())

    cve_match = re.match(r"(?i)^cve[-\s]?(\d{4})[-\s]?(\d{4,7})$", cleaned)
    if cve_match:
        return f"CVE-{cve_match.group(1)}-{cve_match.group(2)}"

    alias = ENTITY_ALIASES.get(cleaned.lower())
    if alias:
        return alias

    return cleaned


# -------------------------------------------
# EXTRACT ENTITIES & RELATIONSHIPS FROM TEXT
# -------------------------------------------
def extract_graph_data(text, source_filename):
    """Use LLM to extract cybersecurity entities and relationships from a text chunk."""

    entity_types_list = ", ".join(sorted(VALID_ENTITY_TYPES))
    relation_types_list = ", ".join(sorted(VALID_RELATIONS))

    prompt = f"""You are a cybersecurity knowledge graph expert. Extract entities and
relationships from the text below, following a strict cybersecurity ontology.

ALLOWED ENTITY TYPES (use exactly one of these for entity1_type / entity2_type):
{entity_types_list}

ALLOWED RELATIONSHIP TYPES (use exactly one of these for "relation" — do NOT invent
new ones, and NEVER use generic relations like RELATED_TO, CONNECTED_WITH, or
ASSOCIATED_WITH):
{relation_types_list}

Return ONLY a JSON array like this (no explanation, no markdown, just raw JSON):
[
  {{"entity1": "CVE-2024-1234", "entity1_type": "Vulnerability", "relation": "AFFECTS", "entity2": "Apache HTTP Server", "entity2_type": "Product", "confidence": 0.9}},
  {{"entity1": "APT29", "entity1_type": "ThreatActor", "relation": "USES", "entity2": "PowerShell", "entity2_type": "AttackTechnique", "confidence": 0.85}},
  {{"entity1": "Emotet", "entity1_type": "Malware", "relation": "COMMUNICATES_WITH", "entity2": "185.220.101.5", "entity2_type": "IPAddress", "confidence": 0.8}},
  {{"entity1": "Rate Limiting", "entity1_type": "SecurityControl", "relation": "MITIGATES", "entity2": "DDoS Attack", "entity2_type": "AttackTechnique", "confidence": 0.85}},
  {{"entity1": "Acme Corp", "entity1_type": "Organization", "relation": "COMPLIES_WITH", "entity2": "ISO 27001", "entity2_type": "Standard", "confidence": 0.75}}
]

Rules:
- Extract at most 12 of the most important relationships
- Keep entity names short and specific (e.g. "CVE-2024-1234", not "a vulnerability")
- Only use entity types and relation types from the allowed lists above
- confidence is your own estimate (0.0-1.0) of how clearly the text supports this triple
- If no clear cybersecurity relationships are found, return an empty array: []

Text:
{text[:2000]}

JSON:"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()

        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        relationships = json.loads(content)
        return relationships
    except Exception as e:
        print(f"⚠️ Entity extraction failed: {e}")
        return []


# -------------------------------------------
# STORE GRAPH DATA IN NEO4J
# -------------------------------------------
def store_in_graph(relationships, source_filename, page=None, chunk_id=None):
    """Store extracted cybersecurity entities and relationships in Neo4j,
    with typed nodes, a controlled relationship vocabulary, normalized
    entity names, and full relationship metadata (source, page, chunk_id,
    confidence, extraction_method)."""

    if not relationships:
        return 0

    stored_count = 0

    try:
        with get_driver().session() as session:
            for rel in relationships:
                entity1_raw = rel.get("entity1", "").strip()
                entity2_raw = rel.get("entity2", "").strip()
                entity1_type = rel.get("entity1_type", "").strip()
                entity2_type = rel.get("entity2_type", "").strip()
                relation = rel.get("relation", "").strip().upper().replace(" ", "_")
                confidence = rel.get("confidence", 0.7)

                if not entity1_raw or not relation or not entity2_raw:
                    continue

                # Reject generic/non-standard relationship types outright,
                # per the issue's technical notes.
                if relation not in VALID_RELATIONS:
                    print(f"⚠️ Skipping non-standard relation type: {relation}")
                    continue

                entity1 = normalize_entity_name(entity1_raw)
                entity2 = normalize_entity_name(entity2_raw)

                # Fall back to a generic "Entity" label if the type wasn't
                # recognized, rather than dropping a real relationship.
                label1 = entity1_type if entity1_type in VALID_ENTITY_TYPES else "Entity"
                label2 = entity2_type if entity2_type in VALID_ENTITY_TYPES else "Entity"

                # Neo4j doesn't allow parameterized labels/relationship types,
                # so we build them into the query string — safe here because
                # label1/label2/relation are always validated against the
                # fixed whitelists above before this point.
                query = f"""
                MERGE (a:{label1} {{name: $entity1}})
                MERGE (b:{label2} {{name: $entity2}})
                MERGE (a)-[r:{relation}]->(b)
                SET r.source = $source,
                    r.page = $page,
                    r.chunk_id = $chunk_id,
                    r.confidence = $confidence,
                    r.extraction_method = 'llm'
                RETURN a, b, r
                """

                session.run(
                    query,
                    entity1=entity1,
                    entity2=entity2,
                    source=source_filename,
                    page=page,
                    chunk_id=chunk_id,
                    confidence=confidence
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
    """Process document chunks and build the cybersecurity knowledge graph."""

    try:
        with get_driver().session() as session:
            result = session.run(
                "MATCH ()-[r]->() WHERE r.source = $source RETURN count(r) as count",
                source=source_filename
            )
            count = result.single()["count"]
            if count > 0:
                print(f"⚡ Graph already exists for {source_filename} ({count} relationships) — skipping!")
                return count
    except Exception as e:
        print(f"⚠️ Graph check error: {e}")

    print(f"\n🔗 Building cybersecurity knowledge graph for: {source_filename}")
    total_relationships = 0

    for i, chunk in enumerate(chunks):
        if i % 3 != 0:
            continue

        text = chunk.page_content
        if len(text.strip()) < 100:
            continue

        page = chunk.metadata.get("page")
        chunk_id = chunk.metadata.get("chunk_index", i)

        print(f"   🧠 Extracting entities from chunk {i+1}/{len(chunks)}...")
        relationships = extract_graph_data(text, source_filename)

        if relationships:
            count = store_in_graph(relationships, source_filename, page=page, chunk_id=chunk_id)
            total_relationships += count

    print(f"✅ Graph built: {total_relationships} total relationships for {source_filename}")
    return total_relationships


# -------------------------------------------
# QUERY GRAPH FOR RELEVANT CONTEXT
# -------------------------------------------
def graph_retrieve(question, top_k=5):
    """Multi-hop graph traversal to find related entities not directly in embeddings.
    Matches any relationship type / entity label, since the graph now uses
    typed nodes and typed relations instead of a single generic RELATION type."""

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
                    MATCH (a)-[r]->(b)
                    WHERE toLower(a.name) CONTAINS toLower($keyword)
                       OR toLower(b.name) CONTAINS toLower($keyword)
                    RETURN a.name as entity1, type(r) as relation, b.name as entity2
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
                    MATCH (a)-[r]->(b)
                    WHERE toLower(a.name) CONTAINS toLower($entity)
                       OR toLower(b.name) CONTAINS toLower($entity)
                    RETURN a.name as entity1, type(r) as relation, b.name as entity2
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
    """Remove all graph data for a specific file. Matches on the source
    property regardless of relationship type, since types now vary
    (AFFECTS, USES, MITIGATES, etc.) instead of a single generic RELATION."""
    try:
        with get_driver().session() as session:
            query = """
            MATCH ()-[r]->()
            WHERE r.source = $source
            DELETE r
            """
            session.run(query, source=source_filename)

            # Clean up orphan nodes (any label)
            session.run("""
            MATCH (n)
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