import * as fs from "fs";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import axios from "axios";
import { indexDocument, generateEmbedding } from "./rag/index.js";
import { connectToVectorDB, openTable, tableSearch } from "./db/index.js";
import { Table } from "vectordb";
import { rankingResponses } from "./rerank/index.js";

// --- Interfaces ---
interface SearchResult {
  id: number;
  text: string;
  _distance?: number;
}

// --- Configuración ---
const OLLAMA_URL = "http://localhost:11434";
const LLM_MODEL = "mistral:7b-instruct";
const COLLECTION_NAME = "documents";
const TEXTCHUNKWIDE = 2;

async function askQuestion(question: string): Promise<void> {
  const db = await connectToVectorDB();
  const table = await openTable(COLLECTION_NAME);
  
  const queryEmbedding = await generateEmbedding(question);

  // En 'vectordb' (JS), .execute() devuelve directamente un Array
  const results = await tableSearch(table, queryEmbedding, 6);
  
  if (results.length === 0) {
    console.log("⚠️ No se encontró información relevante.");
    return;
  }

  const contextChunks = new Set<string>();

  for (const res of results) {
    const idx = Number(res.id);
    if (isNaN(idx)) {
      contextChunks.add(res.text);
      continue;
    }

    const totalRows = await table.countRows();
    const start = Math.max(0, idx - TEXTCHUNKWIDE);
    const end = Math.min(totalRows, idx + TEXTCHUNKWIDE);

    // Búsqueda de vecinos: también devuelve un Array directo
    const neighbors = await table
      .filter(`id >= ${start} AND id <= ${end}`)
      .execute() as SearchResult[];

    neighbors.sort((a, b) => a.id - b.id);
    const enrichedText = neighbors
        .map(n => n.text)
        .join(" "); // O "\n" si prefieres separar los párrafos

    // 4. Añadimos el bloque completo (Texto + Anterior + Posterior) al Set
    if (enrichedText.trim().length > 0) {
      contextChunks.add(enrichedText);
    }
  }
  const reRankedContextChunks = await rankingResponses(question, Array.from(contextChunks));
  const finalContext = Array.from(reRankedContextChunks).join("\n---\n");

  try {
    const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
      model: LLM_MODEL,
      prompt: `Usa el CONTEXTO para responder la PREGUNTA, no te inventes nada ni hagas suposiciones, responde
      con la información contenida en el CONTEXTO!.\n\nCONTEXTO:\n${finalContext}\n\nPREGUNTA: ${question}`,
      stream: false
    });

    console.log("\n🧠 RESPUESTA:\n" + response.data.response);
  } catch (error: any) {
    console.error("❌ Error en Ollama:", error.message);
  }
}

async function main(): Promise<void> {
  try {
    const archivoALeer = "/home/msi/Desktop/AI/RAG/rag-local/texto.txt"; 
    
    if (!fs.existsSync(archivoALeer)) {
      console.error(`❌ El archivo ${archivoALeer} no existe.`);
      return;
    }

    console.log("⏳ Indexando documento...");
    await indexDocument(archivoALeer);
    
    const rl = readline.createInterface({ input, output });
    console.log("\n✅ Sistema listo. Escribe 'salir' para terminar.");

    while (true) {
      const pregunta = await rl.question("\n❓ Haz tu pregunta: ");
      if (["salir", "exit", "quit"].includes(pregunta.toLowerCase())) break;
      if (!pregunta.trim()) continue;

      await askQuestion(pregunta);
    }
    rl.close();
  } catch (error) {
    console.error("🔴 Error crítico:", error);
  }
}

main();


//Mejoras:
//Ajusta modelo, temperatura y ks
//Ajusta el contexto
//Ajusta el ranking, para que si no devuelve respuestas ajustadas, no haya respuesta por parte del modelo => ahorro de costes.


