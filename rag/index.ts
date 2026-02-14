//import pdf from "pdf-parse-fork";
import { chunkText, recursiveChunkingBySentences } from '../chunk/index.ts';
import * as fs from "fs";
import { connectToVectorDB } from "../db/index.ts";
import * as lancedb from "vectordb";
import axios, { AxiosResponse } from "axios";

// --- Interfaces ---
interface EmbeddingResponse {
  embedding: number[];
}

interface VectorRow {
  id: number;
  vector: number[];
  text: string;
  chapter: string;
}

// --- Configuración ---
const EMBEDDING_MODEL: string = "nomic-embed-text";
const OLLAMA_URL: string = "http://localhost:11434";
const COLLECTION_NAME: string = "documents";

export async function indexDocument(path: string): Promise<void> {
  console.log("Extrayendo contenido...");
  const fullText: string = fs.readFileSync(path, 'utf8');

  // 1. Split by chapters and detect the chapter strings. "Capítulo X: Título"
  const chapterRegex = /(Capítulo\s+\d+:?\s*[^\n]+)/gi;
  const sections = fullText.split(chapterRegex);
  
  const rows: VectorRow[] = [];
  let globalChunkIndex = 0;
  let currentChapterTitle = "Introducción / Prólogo";

  console.log("Procesando secciones y generando chunks enriquecidos...");

  for (let i = 0; i < sections.length; i++) {
    const section = sections[i].trim();
    if (!section) continue;

    if (section.match(chapterRegex)) {
      currentChapterTitle = section;
      continue;
    }

    // 2. Chunks by sentences in current chapter
    const subChunks: string[] = recursiveChunkingBySentences(section);

    for (const subChunk of subChunks) {
      const embedding: number[] = await generateEmbedding(subChunk);
      
      if (embedding?.length === 768) {
        rows.push({
          id: globalChunkIndex++,
          vector: Array.from(embedding),
          text: subChunk,
          chapter: currentChapterTitle //add title of chapter as metadata
        });
      }
    }
  }

  const db = await connectToVectorDB();
  const tableNames = await db.tableNames();

  if (tableNames.includes(COLLECTION_NAME)) {
    const table = await db.openTable(COLLECTION_NAME);
    await table.add(rows);
  } else {
    await db.createTable(COLLECTION_NAME, rows);
  }
}

export async function generateEmbedding(text: string): Promise<number[]> {
  try {
    const response: AxiosResponse<EmbeddingResponse> = await axios.post(`${OLLAMA_URL}/api/embeddings`, {
      model: EMBEDDING_MODEL,
      prompt: text
    });
    return response.data.embedding;
  } catch (error: any) {
    console.error("Error while generating embedding.");
    console.error(error);
    throw error;
  }
}