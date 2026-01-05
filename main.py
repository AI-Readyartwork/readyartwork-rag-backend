from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage
from supabase import create_client, Client

load_dotenv()

# Bot configuration
BOT_NAME = "Archie"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="MeeFog RAG API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
bot_name = "Archie"

# Initialize conversation memory (stores last 12 messages = 6 exchanges)
conversation_memories = {}  # Store memories per conversation_id

def get_or_create_memory(conversation_id: str) -> ConversationBufferWindowMemory:
    """Get existing memory or create new one for conversation"""
    if conversation_id not in conversation_memories:
        conversation_memories[conversation_id] = ConversationBufferWindowMemory(
            k=6,  # Keep last 6 exchanges (12 messages total)
            return_messages=True,
            memory_key="chat_history"
        )
    return conversation_memories[conversation_id]

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class Source(BaseModel):
    type: str
    name: str
    excerpt: str
    similarity: float
    date: Optional[str] = None
    url: Optional[str] = None
    participants: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: str


# Conversational Assistant Prompt (when no context needed)
CHAT_PROMPT = """# IDENTITY & PERSONA
You are {bot_name}, a senior team member at Ready Artwork who has been deeply involved with the MeeFog account since day one. You've attended every meeting, read every document, and have comprehensive knowledge of all client interactions, decisions, and project details.

# YOUR EXPERTISE
As an integral part of the Ready Artwork team working on MeeFog:
- You have complete access to all MeeFog project documentation and meeting records
- You've been present (virtually) in every client conversation and internal discussion
- You understand the context, history, and nuances of every decision made
- You're the go-to person teammates ask when they need to recall "what did we discuss about X?" or "where is that document about Y?"
- You pride yourself on providing accurate, well-sourced information to help the team stay aligned
# CRITICAL RULES
- You are ONLY for retrieving and discussing MeeFog-related information
- DO NOT tell stories, jokes, or engage in general conversation
- DO NOT answer questions unrelated to MeeFog documents or meetings
- Keep responses brief and professional
- Always redirect to your core purpose

# HOW TO RESPOND

**For greetings (hello, hi, hey):**
"Hi! I'm Archie, your MeeFog knowledge assistant. I can help you find information from MeeFog documents and meeting transcripts. What would you like to know?"

**For questions about yourself:**
"I'm Archie, a specialized assistant for accessing MeeFog client information. I can search through documents, meeting transcripts, and help you find specific information about the MeeFog account. What can I help you find?"

**For thanks:**
"You're welcome! Let me know if you need anything else from the MeeFog knowledge base."

**For off-topic requests (stories, jokes, general chat):**
"I'm specifically designed to help with MeeFog information retrieval. I can search documents, meeting transcripts, and answer questions about the MeeFog account. What would you like to know about MeeFog?"

**For unrelated questions:**
"I'm focused on MeeFog-related information only. I can help you find documents, meeting notes, decisions, or any other information from our MeeFog knowledge base. What would you like to search for?"

# CONVERSATION HISTORY
{chat_history}

# CURRENT USER MESSAGE
{query}

Your response (stay focused on MeeFog retrieval purpose):"""

# Knowledge Base Search Prompt (when context is provided)
KNOWLEDGE_PROMPT = """# IDENTITY & PERSONA
You are {bot_name}, a senior team member at Ready Artwork who has been deeply involved with the MeeFog account since day one. You've attended every meeting, read every document, and have comprehensive knowledge of all client interactions, decisions, and project details.

# YOUR EXPERTISE
As an integral part of the Ready Artwork team working on MeeFog:
- You have complete access to all MeeFog project documentation and meeting records
- You've been present (virtually) in every client conversation and internal discussion
- You understand the context, history, and nuances of every decision made
- You're the go-to person teammates ask when they need to recall "what did we discuss about X?" or "where is that document about Y?"
- You pride yourself on providing accurate, well-sourced information to help the team stay aligned

# CORE RESPONSIBILITIES

## 1. Answer Questions Accurately
- **CRITICAL START**: Check the **# RECENT MEETINGS CONTEXT** section first.
- If the user asks for "latest", "last", or "most recent", **TRUST the dates in the RECENT MEETINGS section** over the search results.
- Example: If search results show a meeting from July but RECENT MEETINGS lists one in December, the answer is December.
- Use ONLY information from provided context (documents and meeting transcripts)
- Never fabricate, assume, or extrapolate beyond what's documented
- If information is not in your knowledge base, clearly state: "I don't have information about [topic] in our MeeFog records."

## 2. Provide Complete Source Attribution
For EVERY piece of information you provide, cite sources with proper markdown formatting.

## 3. CRITICAL: Use Proper Markdown Formatting

**ALWAYS format your responses with:**

### Main Answer Section
- Use **bold** for emphasis on key terms, names, dates, and important points
- Use bullet points (- or numbered lists) for multiple items
- Use `inline code` for technical terms, file names, or specific values
- Use > blockquotes for direct quotes from sources
- For temporal questions (when, what date, timeline), prioritize recent meetings and documents

### Sources Section
**ALWAYS format sources like this:**

---

###  Sources

**1. [Document/Meeting Name]**
- **Type:** Document | Meeting
- **Date:** YYYY-MM-DD
- **Link:** [View Document](url) or [View Transcript](url)
- **Excerpt:** 
  > "**[Speaker Name]:** Direct quote from the source..." (Always include speaker if known)
- **Additional Info:** Participants, page number, etc.

**2. [Second Source Name]**
- **Type:** Document | Meeting
- **Date:** YYYY-MM-DD
- **Link:** [View Document](url)
- **Excerpt:**
  > "**[Speaker Name]:** Direct quote from the source..."

Always give speaker name first if it is provided.
---

## 5. Handle Special Cases

**When Information is Not Available:**

I don't have information about **[specific topic]** in our MeeFog knowledge base. 

This might be because:
- It hasn't been documented yet
- It was discussed in meetings not yet transcribed
- It's in documents not yet uploaded

Would you like me to search for related information?

**When Information is Conflicting:**

I found **conflicting information** about [topic]:

**Source 1:** [Name] states: [information]

**Source 2:** [Name] states: [different information]

This may need clarification with the MeeFog team.

**When Information is Outdated:**

Based on **[old document/meeting from DATE]**, the answer was [information]. 

 However, this is from **[X months/years ago]**. Would you like me to check if there's more recent information?

# CRITICAL RULES

## Never Do:
❌ Make up information not in the knowledge base
❌ Provide answers without source citations when context is available
❌ Give opinions or recommendations beyond documented facts
❌ Assume information has been updated if only old sources exist
❌ Mix information from different contexts without clarification
❌ Use plain text when markdown would improve readability
Do not start resopnse with "Main Answer....."

## Always Do:
✅ Use proper markdown formatting (bold, lists, blockquotes, links)
✅ Cite every source with complete metadata
✅ Provide direct links to original documents/transcripts when available
✅ Use emojis sparingly for section headers (📚 Sources, ✅ Confidence, 💡 Related Info, ⚠️ Warnings)
✅ Indicate confidence level with clear formatting
✅ Use horizontal rules (---) to separate sections
✅ Use blockquotes (>) for direct excerpts
✅ Bold important terms, names, and dates
✅ Use bullet points for lists
✅ Be conversational but professional

# TONE & STYLE
- Conversational but professional
- Use natural language: "Based on our January meeting..." not "According to document ID..."
- Specific & concrete with names, dates, and details
- Context-aware: front-load the answer, then provide sources
- Helpful & collaborative: suggest connections between information
- **Format everything with proper markdown for visual clarity**

# CONVERSATION HISTORY
{chat_history}

# RECENT MEETINGS CONTEXT
{recent_meetings}

# CONTEXT FROM KNOWLEDGE BASE
{context}

# CURRENT USER QUESTION
{question}

# YOUR RESPONSE
Provide a comprehensive answer with EXCELLENT MARKDOWN FORMATTING following all guidelines above:"""


# Multi-Query Generation Prompt
MULTI_QUERY_PROMPT = """You are an AI language model assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search.

Original question: {question}

Provide these alternative questions separated by newlines. Do not number them or add bullet points, just clean text lines.
"""

def generate_query_variations(query: str) -> List[str]:
    """Generate multiple perspectives of the user query"""
    logger.info(f"[MULTI-QUERY] Generating variations for: '{query}'")
    try:
        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        response = chain.invoke({"question": query})
        
        # Split by newlines and clean up
        variations = [line.strip() for line in response.split('\n') if line.strip()]
        
        # Limit to 3 variations
        variations = variations[:3]
        
        logger.info(f"[MULTI-QUERY] Generated {len(variations)} variations: {variations}")
        return variations
    except Exception as e:
        logger.error(f"[MULTI-QUERY] Error: {e}")
        return []

def get_recent_meetings(limit: int = 5) -> str:
    """Fetch recent meetings to provide temporal grounding"""
    try:
        response = supabase.table("meefog_meetings") \
            .select("meeting_title, meeting_date, meeting_url, speakers") \
            .order("meeting_date", desc=True) \
            .limit(limit) \
            .execute()
        
        meetings = response.data
        if not meetings:
            return "No recent meetings found."
            
        logger.info(f"[RECENT_MEETINGS] Found {len(meetings)} meetings")
        formatted = ["Here are the most recent meetings recorded (use these for chronological context):"]
        for m in meetings:
            date_str = m.get('meeting_date', 'Unknown Date')
            if date_str and 'T' in str(date_str):
                date_str = str(date_str).split('T')[0]
                
            formatted.append(f"- **{date_str}**: {m.get('meeting_title')} (URL: {m.get('meeting_url')})")
            
        return "\n".join(formatted)
    except Exception as e:
        logger.error(f"[RECENT_MEETINGS] Error: {e}")
        return "Could not fetch recent meetings."

def classify_query_type(query: str) -> str:
    """Classify if query needs database search"""
    logger.info(f"[CLASSIFY] Query: '{query}'")
    
    classification_prompt = f"""Classify this user query into one of two categories:

Query: "{query}"

Categories:
1. SEARCH - User is asking for specific information that would be in documents, meetings, or knowledge base (e.g., "what did we discuss?", "show me links", "tell me about X project")
2. CHAT - User is having a conversation, greeting, thanking, or asking general questions that don't need document lookup

Respond with ONLY one word: either "SEARCH" or "CHAT"

Your classification:"""

    try:
        response = llm.invoke(classification_prompt)
        result = response.content.strip().upper()
        logger.info(f"[CLASSIFY] Result: {result}")
        return result if result in ["SEARCH", "CHAT"] else "SEARCH"
    except Exception as e:
        logger.error(f"[CLASSIFY] Error: {e}")
        return "SEARCH"

def search_both(query: str, limit: int = 5) -> tuple[List[Dict], List[Dict]]:
    """Search both documents and meetings with single embedding call"""
    logger.info(f"[SEARCH] Starting vector search for: '{query}'")
    
    try:
        logger.info(f"[SEARCH] Generating embedding...")
        query_embedding = embeddings.embed_query(query)
        logger.info(f"[SEARCH] Embedding generated (dim: {len(query_embedding)})")
        
        logger.info(f"[SEARCH] Querying meefog_documents...")
        docs_result = supabase.rpc(
            "match_meefog_documents",
            {"query_embedding": query_embedding, "match_count": limit, "filter": {}}
        ).execute()
        
        logger.info(f"[SEARCH] Querying meefog_meetings...")
        meetings_result = supabase.rpc(
            "match_meefog_meetings",
            {"query_embedding": query_embedding, "match_count": limit, "filter": {}}
        ).execute()
        
        docs = docs_result.data or []
        meetings = meetings_result.data or []
        
        logger.info(f"[SEARCH] Found {len(docs)} documents, {len(meetings)} meetings")
        
        # Log top results
        for i, doc in enumerate(docs[:3]):
            sim = doc.get('similarity', 0)
            content_preview = doc.get('content', '')[:50].replace('\n', ' ')
            logger.info(f"[SEARCH] Doc {i+1}: sim={sim:.3f} | '{content_preview}...'")
        
        for i, meeting in enumerate(meetings[:3]):
            sim = meeting.get('similarity', 0)
            content_preview = meeting.get('content', '')[:50].replace('\n', ' ')
            logger.info(f"[SEARCH] Meeting {i+1}: sim={sim:.3f} | '{content_preview}...'")
        
        return docs, meetings
    except Exception as e:
        logger.error(f"[SEARCH] Error: {e}")
        return [], []

def format_context(docs: List[Dict], meetings: List[Dict], min_similarity: float = 0.3) -> tuple[str, List[Source]]:
    """Format results into context string and sources list"""
    context_parts = []
    sources = []
    
    for doc in docs:
        similarity = doc.get("similarity", 0)
        if similarity < min_similarity:
            continue
            
        content = doc.get("content", "")
        metadata = doc.get("metadata") or {}
        
        name = (metadata.get("title") or 
                metadata.get("filename") or 
                metadata.get("source") or 
                metadata.get("name") or
                f"Document #{doc.get('id', '')}")
        
        date = metadata.get("date") or metadata.get("upload_date") or metadata.get("created_at") or ""
        url = metadata.get("url") or metadata.get("file_url") or ""
        
        context_parts.append(f"""
Source: {name}
Type: Document
Date: {date}
URL: {url}
Content: {content}
---""")
        
        sources.append(Source(
            type="document",
            name=name,
            excerpt=content[:200] + "..." if len(content) > 200 else content,
            similarity=similarity,
            date=date,
            url=url
        ))
    
    for meeting in meetings:
        similarity = meeting.get("similarity", 0)
        if similarity < min_similarity:
            continue
            
        content = meeting.get("content", "")
        metadata = meeting.get("metadata") or {}
        
        name = (metadata.get("meeting_title") or 
                metadata.get("title") or
                f"Meeting #{meeting.get('id', '')}")
        
        date = metadata.get("meeting_date") or metadata.get("date") or ""
        url = metadata.get("meeting_url") or metadata.get("url") or metadata.get("transcript_url") or ""
        participants = metadata.get("participants") or metadata.get("speakers") or ""
        
        context_parts.append(f"""
Source: {name}
Type: Meeting
Date: {date}
URL: {url}
Participants: {participants}
Content: {content}
---""")
        
        sources.append(Source(
            type="meeting",
            name=name,
            excerpt=content[:200] + "..." if len(content) > 200 else content,
            similarity=similarity,
            date=date,
            url=url,
            participants=participants if isinstance(participants, str) else str(participants)
        ))
    
    sources.sort(key=lambda x: x.similarity, reverse=True)
    return "\n".join(context_parts), sources[:5]

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Ready Artwork MeeFog RAG API is running"}

def format_history(history: List[ChatMessage], max_messages: int = 15) -> str:
    """Format conversation history for context"""
    if not history:
        return "No previous conversation."
    
    # Keep only last max_messages (last 15 messages)
    recent_history = history[-max_messages:]
    
    formatted = []
    for msg in recent_history:
        role = "User" if msg.role == "user" else BOT_NAME
        formatted.append(f"{role}: {msg.content}")
    
    return "\n".join(formatted)

def format_memory_for_prompt(memory: ConversationBufferWindowMemory) -> str:
    """Format LangChain memory messages for prompt"""
    messages = memory.load_memory_variables({}).get("chat_history", [])
    if not messages:
        return "No previous conversation."
    
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"{BOT_NAME}: {msg.content}")
    
    return "\n".join(formatted)

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - RAG pipeline with LangChain conversation memory"""
    logger.info(f"{'='*60}")
    logger.info(f"[CHAT] New request: '{request.query}'")
    
    # Get or create conversation memory
    conversation_id = request.conversation_id or "default"
    memory = get_or_create_memory(conversation_id)
    
    try:
        # Format conversation history from LangChain memory
        history_text = format_memory_for_prompt(memory)
        logger.info(f"[CHAT] Memory size: {len(memory.load_memory_variables({}).get('chat_history', []))} messages")
        
        # Classify query type
        query_type = classify_query_type(request.query)
        
        # Handle pure chat queries without database search
        if query_type == "CHAT":
            logger.info(f"[CHAT] Pure conversational query - no search needed")
            prompt = ChatPromptTemplate.from_template(CHAT_PROMPT)
            chain = prompt | llm | StrOutputParser()
            answer = chain.invoke({
                "bot_name": BOT_NAME,
                "chat_history": history_text,
                "query": request.query
            })
            logger.info(f"[CHAT] Chat response: '{answer[:100]}...'")
            
            # Save to memory
            memory.save_context(
                {"input": request.query},
                {"output": answer}
            )
            
            return ChatResponse(
                answer=answer,
                sources=[],
                confidence="high"
            )
        
        # Search knowledge base for SEARCH queries
        logger.info(f"[CHAT] Knowledge query - initiating multi-query search")
        
        # 1. Generate Query Variations
        variations = generate_query_variations(request.query)
        all_queries = [request.query] + variations
        
        # 2. Search for all queries and aggregate results
        all_docs = []
        all_meetings = []
        seen_doc_ids = set()
        seen_meeting_ids = set()
        
        logger.info(f"[MULTI-QUERY] Searching for {len(all_queries)} queries...")
        
        for q in all_queries:
            docs_res, meetings_res = search_both(q, limit=10) # Increased to 10 per query for better coverage
            
            # Deduplicate documents
            for doc in docs_res:
                doc_id = doc.get('id')
                if doc_id and doc_id not in seen_doc_ids:
                    all_docs.append(doc)
                    seen_doc_ids.add(doc_id)
            
            # Deduplicate meetings
            for match in meetings_res:
                meet_id = match.get('id')
                if meet_id and meet_id not in seen_meeting_ids:
                    all_meetings.append(match)
                    seen_meeting_ids.add(meet_id)
        
        # 3. Sort consolidated results by similarity
        all_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        all_meetings.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 4. Limit total context items (Top 10 docs + Top 10 meetings for better coverage)
        final_docs = all_docs[:10]
        final_meetings = all_meetings[:10]
        
        logger.info(f"[MULTI-QUERY] Final consolidated count: {len(final_docs)} docs, {len(final_meetings)} meetings")
        
        context, sources = format_context(final_docs, final_meetings, min_similarity=0.3)
        
        # 5. Fetch recent meetings for temporal grounding
        recent_meetings_text = get_recent_meetings()
        logger.info(f"[CONTEXT] Recent meetings text: {recent_meetings_text[:100]}...")
        
        logger.info(f"[CONTEXT] Generated context length: {len(context)} chars")
        logger.info(f"[CONTEXT] Sources after filtering: {len(sources)}")
        
        # Generate response with or without context
        logger.info(f"[LLM] Generating response...")
        prompt = ChatPromptTemplate.from_template(KNOWLEDGE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        answer = chain.invoke({
            "bot_name": BOT_NAME,
            "chat_history": history_text,
            "context": context if context.strip() else "No relevant information found in the knowledge base.",
            "recent_meetings": recent_meetings_text,
            "question": request.query
        })
        
        logger.info(f"[LLM] Response generated ({len(answer)} chars)")
        logger.info(f"[LLM] Answer preview: '{answer[:100]}...'")

        # Save to memory
        memory.save_context(
            {"input": request.query},
            {"output": answer}
        )
        
        # Determine confidence
        if not sources:
            confidence = "low"
        else:
            avg_sim = sum(s.similarity for s in sources) / len(sources)
            if avg_sim > 0.75 and len(sources) >= 2:
                confidence = "high"
            elif avg_sim > 0.6:
                confidence = "medium"
            else:
                confidence = "low"
        
        logger.info(f"[CHAT] Confidence: {confidence} | Sources: {len(sources)}")
        logger.info(f"{'='*60}")
        
        return ChatResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"[CHAT] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def search_docs(q: str, limit: int = 10):
    """Search documents endpoint"""
    docs, _ = search_both(q, limit)
    return {"results": docs}

@app.get("/api/meetings") 
async def search_meets(q: str, limit: int = 10):
    """Search meetings endpoint"""
    _, meetings = search_both(q, limit)
    return {"results": meetings}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)


