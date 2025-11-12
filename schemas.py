"""
Database Schemas for VectorTutor

Each Pydantic model represents a collection in MongoDB. The collection name is the
lowercase of the class name.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class User(BaseModel):
    email: str
    name: str
    preferences: Dict[str, Any] = Field(default_factory=dict)

class Document(BaseModel):
    user_id: Optional[str] = None
    title: str
    source_filename: str
    mode: str = Field(default="openai", description="embedding/llm mode: 'openai' | 'local' | 'gemini'")
    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)

class Flashcard(BaseModel):
    user_id: Optional[str] = None
    document_id: str
    topic: Optional[str] = None
    question: str
    answer: str
    tags: List[str] = Field(default_factory=list)

class Quiz(BaseModel):
    user_id: Optional[str] = None
    document_id: str
    topic: Optional[str] = None
    difficulty: str = Field(default="Easy")
    questions: List[Dict[str, Any]]

class QuizResult(BaseModel):
    user_id: Optional[str] = None
    quiz_id: str
    score: float
    answers: List[Dict[str, Any]]
    weak_topics: List[str] = Field(default_factory=list)

class Plan(BaseModel):
    user_id: Optional[str]
    document_id: Optional[str] = None
    schedule: List[Dict[str, Any]]
    generated_at: datetime = Field(default_factory=datetime.utcnow)
