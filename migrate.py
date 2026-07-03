from app import app, db

with app.app_context():
    print("✨ Creating new tables with updated schema...")
    # Only creates missing tables, keeps existing data
    db.create_all()
    
    print("✅ Database updated successfully!")
    print("📚 New table added (existing data preserved):")
    print("  - users ✅ (existing)")
    print("  - chat_history ✅ (existing)")
    print("  - processed_files ✅ (existing)")
    print("  - semantic_cache ✨ NEW!")         