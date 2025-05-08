import os
import json
import re
import numpy as np
import logging
import traceback
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager
from datetime import datetime
from uuid import uuid4

import requests
from dotenv import load_dotenv

import faiss
import torch
import uvicorn
from fastapi import FastAPI, Depends, Header
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi.middleware.cors import CORSMiddleware

# Base de datos con SQLAlchemy
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Carga inicial
load_dotenv()
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
CACHE_DIR = os.getenv("CACHE_DIR", "./cache/")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = os.getenv("HF_DATASETS_CACHE", os.path.join(CACHE_DIR, "hf_datasets"))
os.environ["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE", os.path.join(CACHE_DIR, "hf_transformers"))

# ORM setup
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./chat_history.db")
engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
class Message(Base):
    __tablename__ = "messages"
    id = Column(String, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String, index=True)
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
Base.metadata.create_all(bind=engine)

# Configuración global
class Config:
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
    DATA_DIR = os.getenv("DATA_DIR", "./data/")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))
    TOP_K = int(os.getenv("TOP_K", 3))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.85))
    NON_MAIN_TOPIC_PENALTY = float(os.getenv("NON_MAIN_TOPIC_PENALTY", 0.8))
    GENERATION_THRESHOLD = float(os.getenv("GENERATION_THRESHOLD", 0.65))
    SYSTEM_INSTRUCTIONS = os.getenv("SYSTEM_INSTRUCTIONS", "Soy un asistente legal. Responde a las preguntas de manera profesional y precisa.")
    API_KEY = os.getenv("API_KEY")
    MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    HISTORY_WINDOW = 5

# Estado app
class AppState:
    def __init__(self):
        self.embedding_model = None
        self.predefined_index = None
        self.predefined_docs = []
        self.rag_index = None
        self.rag_docs = []
        self.user_facts_index = None
        self.user_facts = []
app_state = AppState()

# Pydantic models
class ChatRequest(BaseModel): message: str; main_topic: Optional[str] = ""
class ChatResponse(BaseModel): response: str; source: str; confidence: Optional[float]; document: Optional[str]; article: Optional[str]; topic: Optional[str]; main_topic: Optional[str]

# DB utils
def get_db():
    db = SessionLocal();
    try: yield db
    finally: db.close()
async def save_message(session_id, role, content, db):
    msg = Message(id=str(uuid4()), session_id=session_id, role=role, content=content)
    db.add(msg); db.commit()
async def get_recent_history(session_id, db, limit=Config.HISTORY_WINDOW):
    msgs = db.query(Message).filter(Message.session_id==session_id).order_by(Message.timestamp.desc()).limit(limit).all()
    msgs = list(reversed(msgs))
    return [{"role":m.role,"content":m.content} for m in msgs]

def extract_article_number(text: str) -> Optional[Dict]:
    """
    Extrae el artículo de un texto, p.ej. 'Artículo 12-b de la ley 5'.
    Retorna dict con keys 'article', 'number', 'special', 'reference' o None.
    """
    pattern = re.compile(
        r'\b(?:art[íi]culo?|art\.?)\.?\s*'            # prefijo artículo
        r'(?P<number>\d+[a-zA-Z\-]*)'                 # número (con sufijos opcionales)
        r'(?:\s+de\s+(?P<special>ley|reglamento|norma|código)\s*'
        r'(?P<reference>[\d\-]+))?',                  # referencia opcional (solo números y guiones)
        re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return None

    return {
        'article':   m.group(0).strip(),
        'number':    m.group('number'),
        'special':   (m.group('special') or "").lower() or None,
        'reference': m.group('reference') or None
    }

def load_and_process_data():
    responses, rag_documents = {}, []
    file_count = 0
    for filename in os.listdir(Config.DATA_DIR):
        if filename.endswith(".json"):
            path = os.path.join(Config.DATA_DIR, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                main_topic = data.get("main_topic", "General")
                for intent in data.get("intents", []):
                    tag = intent["tag"]
                    topic = intent.get("topic", "General")
                    meta = {"document": main_topic, "article": intent.get("article","NA"),
                            "tag": tag, "topic": topic, "main_topic": main_topic}
                    responses[tag] = {"patterns": intent["patterns"], "responses": intent["responses"], "metadata": meta}
                    text = (f"Documento: {main_topic}\nArtículo: {meta['article']}\n"
                            f"Preguntas: {', '.join(intent['patterns'])}\n"
                            f"Respuestas: {' '.join(intent['responses'])}")
                    rag_documents.append({"text": text, "metadata": meta})
                file_count += 1
            except Exception as e:
                logger.error(f"Error procesando {filename}: {e}")
    logger.info(f"Carga completada. {file_count} archivos procesados.")
    return responses, rag_documents

def build_index(documents: List[Dict]) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    if not documents:
        return None, []
    texts = [doc["text"] for doc in documents]
    embeddings = app_state.embedding_model.encode(texts, normalize_embeddings=True)
    if embeddings.ndim == 1:
        embeddings = np.expand_dims(embeddings, 0)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    return index, documents


def get_common_document_prefixes() -> List[str]:
    prefixes = set()
    for filename in os.listdir(Config.DATA_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(Config.DATA_DIR, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                if "main_topic" in data:
                    first_word = data["main_topic"].split()[0]
                    prefixes.add(first_word)
    return list(prefixes)

def extract_main_topic_from_message(message: str) -> Tuple[str, str]:
    logger.debug("🔍 Intentando extraer main_topic")
    common_prefixes = get_common_document_prefixes()
    patterns = [
        r"(?:el\s+)?(" + "|".join(re.escape(p) for p in common_prefixes) + r")\s+(?:de|general\s+de)\s+([^\.,\?]+)",
        r"(segun el|de acuerdo al|conforme al|basado en el|como indica el|en el|del|sobre el|acerca del)\s+(.+?)(?:\\.|\\?|$)"
    ]
    extracted_topic = ""
    cleaned_message = message
    for pattern in patterns:
        matches = re.finditer(pattern, message, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                prefix = match.group(1).strip()
                topic_body = match.group(2).strip()
                potential_topic = f"{prefix} {topic_body}"
                logger.debug(f"🔍 Encontrado potencial topic: '{potential_topic}'")
                if len(potential_topic) > 10:
                    extracted_topic = potential_topic.title()
                    cleaned_message = message.replace(match.group(0), "").strip()
                    logger.info(f"✅ Topic extraído: '{extracted_topic}'")
                    return extracted_topic, cleaned_message
    logger.debug("🔎 No se encontró ningún main_topic en el mensaje")
    return "", message

def determine_best_main_topic(extracted_topic: str, explicit_topic: str, message: str) -> str:
    if not explicit_topic:
        return extracted_topic if extracted_topic else "General"
    if not extracted_topic:
        return explicit_topic
    message_embed = app_state.embedding_model.encode(message, normalize_embeddings=True)
    extracted_embed = app_state.embedding_model.encode(extracted_topic, normalize_embeddings=True)
    explicit_embed = app_state.embedding_model.encode(explicit_topic, normalize_embeddings=True)
    sim_extracted = util.pytorch_cos_sim(message_embed, extracted_embed).item()
    sim_explicit = util.pytorch_cos_sim(message_embed, explicit_embed).item()
    logger.debug(f"📊 Similitudes - Extraído: {sim_extracted:.2f}, Explícito: {sim_explicit:.2f}")
    return extracted_topic if sim_extracted > sim_explicit else explicit_topic

def find_matching_main_topic(extracted_topic: str, available_topics: List[str]) -> str:
    logger.debug(f"🔍 Buscando coincidencia para topic extraído: '{extracted_topic}'")
    logger.debug(f"📋 Topics disponibles: {available_topics}")
    if not extracted_topic or not available_topics:
        logger.debug("⚠️ No hay topic extraído o lista de topics disponibles vacía")
        return ""
    try:
        topic_embed = app_state.embedding_model.encode(extracted_topic, normalize_embeddings=True)
        available_embeds = app_state.embedding_model.encode(available_topics, normalize_embeddings=True)
        similarities = util.pytorch_cos_sim(topic_embed, available_embeds)[0]
        for i, (topic, sim) in enumerate(zip(available_topics, similarities)):
            logger.debug(f"📊 Similitud con '{topic}': {sim:.4f}")
        best_idx = torch.argmax(similarities).item()
        best_score = similarities[best_idx].item()
        best_topic = available_topics[best_idx]
        logger.debug(f"🏆 Mejor coincidencia: '{best_topic}' con score {best_score:.4f}")
        if best_score >= 0.85:
            logger.info(f"✅ Match encontrado: '{best_topic}' (Score: {best_score:.4f})")
            return best_topic
        else:
            logger.info(f"⚠️ Score insuficiente para coincidencia: {best_score:.4f} < 0.85")
    except Exception as e:
        logger.error(f"❌ Error calculando similitud: {str(e)}")
        logger.error(traceback.format_exc())
    return ""

def normalize_article(article_str: str) -> str:
    # Convertir a minúsculas y quitar caracteres no numéricos
    article_str = article_str.lower().strip()
    
    # Extraer número y posible sufijo (ej: 34, 45-bis, 12a)
    match = re.search(r'(?:art[íi]culo\s*)?(\d+[\-a-z]*)', article_str)
    return match.group(1) if match else ""

def build_structured_prompt(user_prompt: str, context: str) -> str:
    return (
        Config.SYSTEM_INSTRUCTIONS +
        "\n\nCONTEXTO:\n" +
        context +
        "\n\nPREGUNTA DEL USUARIO:\n" +
        user_prompt
    )

async def call_gemini_api(prompt: str) -> Optional[str]:
    
    logger.debug(prompt)

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {"Content-Type": "application/json"}
    try:
        logger.debug("🔄 Enviando solicitud a API de generación")
        r = requests.post(Config.API_URL, json=payload, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text")
                return text or candidates[0].get("output")
        else:
            logger.error(f"❌ Error API: {r.status_code} {r.text}")
    except Exception as e:
        logger.error(f"❌ Exception calling API: {e}", exc_info=True)
    return None

def hybrid_retrieval(query: str, main_topic: str = None) -> Dict:
    logger.debug(f"\n🔎 Nueva consulta: '{query}'")

    # ——————————————————————————————————————————————————————————————
    # 0) BUSCAR COINCIDENCIA EXACTA EN LOS PATRONES
    # ——————————————————————————————————————————————————————————————
    q_norm = query.strip().lower().rstrip('?.¡!')  # normalizamos
    for tag, intent in app_state.predefined_responses.items():
        for pat in intent["patterns"]:
            if pat.strip().lower().rstrip('?.¡!') == q_norm:
                # Si hay match exacto, devolvemos la mejor respuesta de ese intent
                best = max(
                    intent["responses"],
                    key=lambda r: util.pytorch_cos_sim(
                        app_state.embedding_model.encode(r, normalize_embeddings=True),
                        app_state.embedding_model.encode(query, normalize_embeddings=True)
                    ).item()
                )
                logger.info(f"🎯 Match exacto patrones en '{tag}', devolviendo respuesta predefinida.")
                return {
                    "text":     best,
                    "score":    1.0,
                    "source":   "predefined",
                    "metadata": intent["metadata"]
                }

    # ——————————————————————————————————————————————————————————————
    # 1) Embeddings de la consulta
    # ——————————————————————————————————————————————————————————————
    try:
        query_embed = app_state.embedding_model.encode(query, normalize_embeddings=True)
        query_embed = query_embed.reshape(1, -1).astype('float32')
    except Exception as e:
        logger.error(f"❌ Error embed query: {e}")
        return {"source": "error"}

    # ——————————————————————————————————————————————————————————————
    # 2) Búsqueda FAISS: predefinidas y RAG
    # ——————————————————————————————————————————————————————————————
    predefined_scores, predefined_indices = app_state.predefined_index.search(query_embed, Config.TOP_K * 3)
    rag_scores, rag_indices                   = app_state.rag_index.search(query_embed, Config.TOP_K)

    article_info = extract_article_number(query)
    if article_info and main_topic:
        query_article_num = article_info['number']
        logger.debug(f"🔍 Artículo detectado en la consulta: {query_article_num}")
        # Filtramos candidatos: buscamos en la metadata de cada candidato la coincidencia del artículo Y el main_topic
        filtered_candidates = []
        for i, score in zip(predefined_indices[0], predefined_scores[0]):
            candidate_article = app_state.predefined_docs[i]["metadata"].get("article", "NA")
            candidate_main_topic = app_state.predefined_docs[i]["metadata"].get("main_topic", "")
            normalized_candidate = normalize_article(candidate_article)
            
            # Calcular similitud entre los main_topics
            main_topic_sim = 0
            if candidate_main_topic and main_topic:
                main_topic_sim = util.pytorch_cos_sim(
                    app_state.embedding_model.encode(main_topic, normalize_embeddings=True),
                    app_state.embedding_model.encode(candidate_main_topic, normalize_embeddings=True)
                ).item()
                
            # Solo agregar si el artículo coincide Y el main_topic tiene buena similitud
            if normalized_candidate == query_article_num and main_topic_sim >= 0.80:
                logger.debug(f"✓ Candidato artículo {query_article_num} encontrado en '{candidate_main_topic}' (sim: {main_topic_sim:.2f})")
                filtered_candidates.append((i, score, main_topic_sim))
                
        if filtered_candidates:
            # Ordenar por una combinación de score semántico y similitud de main_topic
            filtered_candidates.sort(key=lambda x: x[1] * 0.6 + x[2] * 0.4, reverse=True)
            best_idx, best_score, topic_sim = filtered_candidates[0]
            
            # La coincidencia de artículo debe ser realmente buena para usarla
            if best_score >= Config.SIMILARITY_THRESHOLD:
                best_predefined = {
                    "text": app_state.predefined_docs[best_idx]["text"],
                    "score": float(best_score),
                    "source": "predefined",
                    "metadata": app_state.predefined_docs[best_idx]["metadata"]
                }
                logger.info(f"✅ Predefinido por artículo encontrado en main_topic correcto: " 
                            f"{best_predefined['metadata']['tag']} (Score: {best_score:.2f}, Topic sim: {topic_sim:.2f})")
                return best_predefined
            else:
                logger.debug(f"⚠️ Artículo encontrado pero score bajo: {best_score:.2f} < {Config.SIMILARITY_THRESHOLD}")
        else:
            logger.debug(f"⚠️ Se encontró artículo {query_article_num} pero no en el main_topic solicitado: '{main_topic}'")
    elif article_info:
        # Si hay artículo pero no main_topic
        query_article_num = article_info['number']
        logger.debug(f"🔍 Artículo detectado en la consulta sin main_topic específico: {query_article_num}")
        filtered_candidates = []
        for i, score in zip(predefined_indices[0], predefined_scores[0]):
            candidate_article = app_state.predefined_docs[i]["metadata"].get("article", "NA")
            normalized_candidate = normalize_article(candidate_article)
            if normalized_candidate == query_article_num:
                filtered_candidates.append((i, score))
        if filtered_candidates:
            filtered_candidates.sort(key=lambda x: x[1], reverse=True)
            best_idx, best_score = filtered_candidates[0]
            
            # También aplicamos umbral de calidad aquí
            if best_score >= Config.SIMILARITY_THRESHOLD:
                best_predefined = {
                    "text": app_state.predefined_docs[best_idx]["text"],
                    "score": float(best_score),
                    "source": "predefined",
                    "metadata": app_state.predefined_docs[best_idx]["metadata"]
                }
                logger.info(f"✅ Predefinido por artículo encontrado: {best_predefined['metadata']['tag']} (Score: {best_predefined['score']:.2f})")
                return best_predefined
            else:
                logger.debug(f"⚠️ Artículo encontrado pero score bajo: {best_score:.2f} < {Config.SIMILARITY_THRESHOLD}")

    # ——————————————————————————————————————————————————————————————
    # 3) Filtrado POR main_topic con penalización atenuada
    # ——————————————————————————————————————————————————————————————
    best_predefined = None
    if main_topic:
        logger.debug(f"🔍 Filtrando por main_topic: '{main_topic}' (penalty={Config.NON_MAIN_TOPIC_PENALTY})")
        candidates = []
        for i, score in zip(predefined_indices[0], predefined_scores[0]):
            meta           = app_state.predefined_docs[i]["metadata"]
            doc_topic      = meta.get("main_topic", "")
            topic_sim      = util.pytorch_cos_sim(
                app_state.embedding_model.encode(main_topic, normalize_embeddings=True),
                app_state.embedding_model.encode(doc_topic, normalize_embeddings=True)
            ).item()

            # 1) excepción: scores ≳0.95 no se penalizan
            if score > 0.95:
                penalty = 1.0
            # 2) si topic_sim <0.8, penalizamos suavemente
            elif topic_sim < 0.8:
                penalty = Config.NON_MAIN_TOPIC_PENALTY
            else:
                penalty = 1.0

            adj_score = score * penalty
            logger.debug(f"[{i}] orig={score:.2f}, sim={topic_sim:.2f}, pen={penalty:.2f} → adj={adj_score:.2f}")

            if adj_score > Config.SIMILARITY_THRESHOLD:
                candidates.append((i, adj_score, topic_sim))

        if candidates:
            # ordenamos por combinación
            candidates.sort(key=lambda x: x[1], reverse=True)
            idx, sc, _ = candidates[0]
            best_predefined = {
                "text":     app_state.predefined_docs[idx]["text"],
                "score":    float(sc),
                "source":   "predefined",
                "metadata": app_state.predefined_docs[idx]["metadata"]
            }
            logger.info(f"✅ Predefinido elegido: {best_predefined['metadata']['tag']} (adj={sc:.2f})")

    # ——————————————————————————————————————————————————————————————
    # 4) Caer a búsqueda general si aún no hay predefinido
    # ——————————————————————————————————————————————————————————————
    if not best_predefined:
        for i, score in zip(predefined_indices[0], predefined_scores[0]):
            if score > Config.SIMILARITY_THRESHOLD:
                best_predefined = {
                    "text":     app_state.predefined_docs[i]["text"],
                    "score":    float(score),
                    "source":   "predefined",
                    "metadata": app_state.predefined_docs[i]["metadata"]
                }
                logger.info(f"✅ Predefinido por score general: {best_predefined['metadata']['tag']} ({score:.2f})")
                break

    # ——————————————————————————————————————————————————————————————
    # 5) Recoger candidatos RAG (igual que antes)
    # ——————————————————————————————————————————————————————————————
    rag_candidates = []
    for i, score in zip(rag_indices[0], rag_scores[0]):
        if score > Config.GENERATION_THRESHOLD * 0.9:
            rag_candidates.append({
                "text":     app_state.rag_docs[i]["text"],
                "score":    float(score),
                "metadata": app_state.rag_docs[i]["metadata"]
            })

    # ——————————————————————————————————————————————————————————————
    # 6) Preferencia entre predefinido y RAG (igual que antes)
    # ——————————————————————————————————————————————————————————————
    if best_predefined and rag_candidates:
        if best_predefined["score"] > Config.SIMILARITY_THRESHOLD * 1.1:
            return best_predefined
        top_rag = max(rag_candidates, key=lambda x: x["score"])
        if top_rag["score"] > best_predefined["score"] * 0.9:
            return {"source": "rag", "rag_candidates": rag_candidates}
        return best_predefined

    if rag_candidates:
        return {"source": "rag", "rag_candidates": rag_candidates}
    if best_predefined:
        return best_predefined

    logger.warning("⚠️ No se encontraron coincidencias suficientes")
    return {"source": "generated"}

# Lifespan: load models & indices
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Iniciando...")
    app_state.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
    # load data
    predefined, rag_docs = load_and_process_data()
    texts = [{"text":p, "metadata":m} for tag,d in predefined.items() for p in d["patterns"] for m in [d["metadata"]]]
    idx_dir = os.path.join(CACHE_DIR, "faiss_indices"); os.makedirs(idx_dir, exist_ok=True)
    def lob(name, docs):
        path=os.path.join(idx_dir,f"{name}.index");
        if os.path.exists(path): idx=faiss.read_index(path)
        else: idx,_=build_index(docs); faiss.write_index(idx,path)
        return idx,docs
    app_state.predefined_index, app_state.predefined_docs=lob("predefined",texts)
    app_state.predefined_responses=predefined
    app_state.rag_index, app_state.rag_docs=lob("rag",rag_docs)
    # user facts index
    dim=app_state.embedding_model.get_sentence_embedding_dimension()
    app_state.user_facts_index=faiss.IndexFlatIP(dim)
    yield
    logger.info("Apagando...")

app=FastAPI(version="0.10.1",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    db=Depends(get_db),
    x_session_id: Optional[str] = Header(None)
):
    # 1. Identificar o crear session_id
    session_id = x_session_id or str(uuid4())
    # 2. Guardar mensaje del usuario
    await save_message(session_id, "user", request.message, db)

    # 3. Recuperar historial reciente
    history = await get_recent_history(session_id, db)
    history_text = "".join(f"{m['role'].title()}: {m['content']}\n" for m in history)

    # 4. Extraer y almacenar factos (ej. email)
    if em := re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", request.message):
        fact = f"Email: {em.group(0)}"
        emb = app_state.embedding_model.encode(fact, normalize_embeddings=True)
        app_state.user_facts_index.add(np.array([emb], dtype="float32"))
        app_state.user_facts.append({"text": fact})

    # 5. Recuperar hechos relevantes
    facts_text = ""
    if app_state.user_facts:
        q_emb = app_state.embedding_model.encode(request.message, normalize_embeddings=True)
        D, I = app_state.user_facts_index.search(np.array([q_emb], dtype="float32"), 3)
        rel = [app_state.user_facts[i]["text"] for i, s in zip(I[0], D[0]) if s > 0.7]
        if rel:
            facts_text = "Datos:\n" + "\n".join(rel) + "\n"

    # 6. Determinar tema principal
    ext, cleaned = extract_main_topic_from_message(request.message)
    eff = determine_best_main_topic(ext, request.main_topic, cleaned if ext else request.message)
    query = cleaned if ext else request.message

    # 7. Recuperación híbrida (RAG/predefinidas)
    result = hybrid_retrieval(query, eff)

    # Caso predefinido (sin cambios)
    if result.get("source") == "predefined":
        tag = result["metadata"]["tag"]
        responses = app_state.predefined_responses[tag]["responses"]
        best = max(
            responses,
            key=lambda r: util.pytorch_cos_sim(
                app_state.embedding_model.encode(r, normalize_embeddings=True),
                app_state.embedding_model.encode(request.message, normalize_embeddings=True)
            ).item()
        )
        await save_message(session_id, "assistant", best, db)
        return ChatResponse(
            response=best,
            source="predefined",
            confidence=result.get("score"),
            document=result["metadata"]["document"],
            article=result["metadata"]["article"],
            topic=result["metadata"]["topic"],
            main_topic=eff
        )

    # Nuevo: caso RAG con múltiples candidatos
    if result.get("source") == "rag":
        rag_candidates = result["rag_candidates"]  # lista de dicts {text, score, metadata}

        # Combina todos los fragmentos de contexto separados por un delimitador
        combined_context = "\n---\n".join(c["text"] for c in rag_candidates)

        # Construye el prompt con todo el contexto RAG
        full_context = history_text + facts_text + combined_context
        structured_prompt = build_structured_prompt(request.message, full_context)

        # Llamada a Gemini con el contexto ampliado
        gen = await call_gemini_api(structured_prompt)
        final_text = gen or "🤔 No pude responder con la información disponible."
        src = "IA-extendida" if gen else "fallback"

        # Guarda la respuesta en historial
        await save_message(session_id, "assistant", final_text, db)

        return ChatResponse(
            response=final_text,
            source=src,
            confidence=None,
            document="✨ Respuesta generada con LIA ✨",
            article=None,
            topic=None,
            main_topic=eff
        )

    # 8. Construir contexto y prompt estructurado
    context = result.get("text", "")
    if art := extract_article_number(query):
        context += f"\nArtículo: {art['article']}"
    full_context = history_text + facts_text + context
    structured_prompt = build_structured_prompt(request.message, full_context)

    # 9. Llamar a Gemini
    gen = await call_gemini_api(structured_prompt)
    src = "IA-extendida" if gen else "fallback"
    final_text = gen or "🤔 No pude responder a esa pregunta con la info disponible."

    # 10. Guardar y devolver
    await save_message(session_id, "assistant", final_text, db)
    return ChatResponse(
        response=final_text,
        source=src,
        confidence=result.get("score"),
        document=result.get("metadata", {}).get("document"),
        article=result.get("metadata", {}).get("article"),
        topic=result.get("metadata", {}).get("topic"),
        main_topic=eff
    )

if __name__=="__main__":
    uvicorn.run("main_with_memory:app",host="0.0.0.0",port=8000,workers=1)
    