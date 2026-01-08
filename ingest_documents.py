"""
Document Ingestion for Archie RAG
Handles document upload, chunking, embedding, and storage in Supabase
Supports: .txt, .pdf, .docx, .doc, .md files
"""

import os
import logging
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema import Document as LangchainDocument
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize clients
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


class DocumentIngestion:
    """Handle document ingestion pipeline"""
    
    def __init__(self):
        self.supabase = supabase
        self.embeddings = embeddings
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.table_name = os.getenv("SUPABASE_TABLE_NAME", "meefog_documents")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"[CONFIG] Chunk size: {self.chunk_size}, Overlap: {self.chunk_overlap}")
        logger.info(f"[CONFIG] Table: {self.table_name}")
    
    def load_document(self, file_path: str) -> List[LangchainDocument]:
        """Load document based on file type"""
        logger.info(f"[LOAD] Loading document: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.txt':
                loader = TextLoader(file_path)
            elif file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            documents = loader.load()
            logger.info(f"[LOAD] Loaded {len(documents)} document(s)")
            return documents
        
        except Exception as e:
            logger.error(f"[LOAD] Error loading document: {e}")
            raise
    
    def chunk_documents(self, documents: List[LangchainDocument]) -> List[LangchainDocument]:
        """Split documents into chunks"""
        logger.info(f"[CHUNK] Chunking {len(documents)} document(s)")
        
        try:
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"[CHUNK] Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"[CHUNK] Error chunking documents: {e}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        logger.info(f"[EMBED] Generating embeddings for {len(texts)} chunks")
        
        try:
            embeddings_list = self.embeddings.embed_documents(texts)
            logger.info(f"[EMBED] Generated {len(embeddings_list)} embeddings")
            return embeddings_list
        except Exception as e:
            logger.error(f"[EMBED] Error generating embeddings: {e}")
            raise
    
    def store_in_supabase(
        self,
        chunks: List[LangchainDocument],
        embeddings_list: List[List[float]],
        metadata: Optional[Dict] = None
    ) -> List[int]:
        """Store chunks and embeddings in Supabase"""
        logger.info(f"[STORE] Storing {len(chunks)} chunks in Supabase")
        
        stored_ids = []
        
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
                # Prepare metadata
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "upload_date": datetime.now().isoformat(),
                }
                
                # Add source information
                if hasattr(chunk, 'metadata'):
                    chunk_metadata.update(chunk.metadata)
                
                # Add custom metadata if provided
                if metadata:
                    chunk_metadata.update(metadata)
                
                # Prepare data for insertion
                data = {
                    "content": chunk.page_content,
                    "metadata": chunk_metadata,
                    "embedding": embedding
                }
                
                # Insert into Supabase
                result = self.supabase.table(self.table_name).insert(data).execute()
                
                if result.data:
                    doc_id = result.data[0]['id']
                    stored_ids.append(doc_id)
                    logger.info(f"[STORE] Stored chunk {i+1}/{len(chunks)} with ID: {doc_id}")
                else:
                    logger.warning(f"[STORE] Failed to store chunk {i+1}")
            
            logger.info(f"[STORE] Successfully stored {len(stored_ids)} chunks")
            return stored_ids
        
        except Exception as e:
            logger.error(f"[STORE] Error storing in Supabase: {e}")
            raise
    
    def ingest_file(
        self,
        file_path: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Complete ingestion pipeline for a single file"""
        logger.info(f"{'='*60}")
        logger.info(f"[INGEST] Starting ingestion for: {file_path}")
        
        try:
            # Step 1: Load document
            documents = self.load_document(file_path)
            
            # Step 2: Chunk documents
            chunks = self.chunk_documents(documents)
            
            # Step 3: Generate embeddings
            texts = [chunk.page_content for chunk in chunks]
            embeddings_list = self.generate_embeddings(texts)
            
            # Step 4: Store in Supabase
            stored_ids = self.store_in_supabase(chunks, embeddings_list, metadata)
            
            result = {
                "success": True,
                "file": file_path,
                "chunks_created": len(chunks),
                "chunks_stored": len(stored_ids),
                "document_ids": stored_ids
            }
            
            logger.info(f"[INGEST] ✅ Successfully ingested: {file_path}")
            logger.info(f"[INGEST] Created {len(chunks)} chunks, stored {len(stored_ids)} in database")
            logger.info(f"{'='*60}")
            
            return result
        
        except Exception as e:
            logger.error(f"[INGEST] ❌ Failed to ingest {file_path}: {e}")
            return {
                "success": False,
                "file": file_path,
                "error": str(e)
            }
    
    def ingest_text(
        self,
        text: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """Ingest raw text directly"""
        logger.info(f"[INGEST] Ingesting raw text ({len(text)} characters)")
        
        try:
            # Create a document object
            doc = LangchainDocument(page_content=text, metadata=metadata or {})
            
            # Chunk the text
            chunks = self.text_splitter.split_documents([doc])
            
            # Generate embeddings
            texts = [chunk.page_content for chunk in chunks]
            embeddings_list = self.generate_embeddings(texts)
            
            # Store in Supabase
            stored_ids = self.store_in_supabase(chunks, embeddings_list, metadata)
            
            result = {
                "success": True,
                "chunks_created": len(chunks),
                "chunks_stored": len(stored_ids),
                "document_ids": stored_ids
            }
            
            logger.info(f"[INGEST] ✅ Successfully ingested text")
            return result
        
        except Exception as e:
            logger.error(f"[INGEST] ❌ Failed to ingest text: {e}")
            return {
                "success": False,
                "error": str(e)
            }
