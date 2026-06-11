from app import app, db

with app.app_context():
    print("✨ Creating new tables with updated schema...")
    # DON'T drop old tables, just create new ones
    db.create_all()  # Only creates missing tables
    
    print("✅ Database updated successfully!")
    print("📚 New tables added (existing data preserved):")
    print("  - semantic_cache ✨ NEW!")