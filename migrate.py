from app import app, db

with app.app_context():
    print("🗑️ Dropping old tables...")
    db.drop_all()
    
    print("✨ Creating new tables with updated schema...")
    db.create_all()
    
    print("✅ Database migrated successfully!")
    print("📚 All tables recreated with user_id column!")