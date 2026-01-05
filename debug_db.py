import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

print("--- Querying Top 5 Recent Meetings ---")
try:
    response = supabase.table("meefog_meetings") \
        .select("id, meeting_title, meeting_date, created_at") \
        .order("meeting_date", desc=True) \
        .limit(5) \
        .execute()
    
    meetings = response.data
    for m in meetings:
        print(f"ID: {m.get('id')} | Date: {m.get('meeting_date')} | Title: {m.get('meeting_title')}")
        
except Exception as e:
    print(f"Error: {e}")
