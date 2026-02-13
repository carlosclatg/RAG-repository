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
  vector: number[]; // LanceDB reconocerá esto como un Float32Array internamente
  text: string;
}

// --- Configuración ---
const EMBEDDING_MODEL: string = "nomic-embed-text";
const OLLAMA_URL: string = "http://localhost:11434";
const COLLECTION_NAME: string = "documents";

/**
 * Procesa un documento, genera embeddings y los guarda en la base de datos.
 */
export async function indexDocument(path: string): Promise<void> {
  console.log("📄 Extrayendo contenido del documento...");
  
  // Nota: Si el archivo es TXT usamos readFileSync, si es PDF usamos extractPDF
  let text: string = ""
  if (path.endsWith('.pdf')) {
    //text = await extractPDF(path);
  } else {
    text = fs.readFileSync(path, 'utf8');
  }
  
  // Chunks por frases
  console.log("Dividiendo en fragmentos (chunks)...");
  const chunks: string[] = recursiveChunkingBySentences(text);

  console.log("Conectando a LanceDB...");
  const db: lancedb.Connection = await connectToVectorDB();

  const rows: VectorRow[] = [];
  
  for (let i = 0; i < chunks.length; i++) {
    const embedding: number[] = await generateEmbedding(chunks[i]);
    
    // Validación de dimensiones (Nomic suele ser 768)
    if (embedding?.length !== 768) {
      console.warn(`⚠️ Embedding omitido en índice ${i}: dimensión incorrecta (${embedding?.length})`);
      continue;
    }

    rows.push({
      id: i,
      vector: Array.from(embedding), // Nos aseguramos de que sea un array de números puro
      text: chunks[i]
    });
  }
  
  console.log(`\n✅ ${rows.length} Embeddings generados.`);

  const tableNames: string[] = await db.tableNames();
  let table: lancedb.Table;

  if (tableNames.includes(COLLECTION_NAME)) {
    console.log("➕ La tabla ya existe. (Lógica de añadir comentada)");
    // table = await db.openTable(COLLECTION_NAME);
    // await table.add(rows);
  } else {
    console.log("🆕 Creando nueva tabla...");
    table = await db.createTable(COLLECTION_NAME, rows);
  }
}

/**
 * Genera el vector numérico de un texto usando Ollama.
 */
export async function generateEmbedding(text: string): Promise<number[]> {
  try {
    const response: AxiosResponse<EmbeddingResponse> = await axios.post(`${OLLAMA_URL}/api/embeddings`, {
      model: EMBEDDING_MODEL,
      prompt: text
    });
    return response.data.embedding;
  } catch (error: any) {
    console.error("❌ Error al generar embedding. ¿Está Ollama corriendo?");
    throw error;
  }
}

/**
 * Extrae texto de un archivo PDF.
 */
// async function extractPDF(path: string): Promise<string> {
//   try {
//     const buffer: Buffer = fs.readFileSync(path);
//     const data: any = await pdf(buffer); // pdf-parse no suele tener tipos oficiales
//     return data.text;
//   } catch (error: any) {
//     console.error("❌ Error al leer el PDF:", error.message);
//     throw error;
//   }
// }