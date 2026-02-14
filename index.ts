import * as fs from "fs";
import * as readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import axios from "axios";
import { indexDocument, generateEmbedding } from "./rag/index.js";
import { connectToVectorDB } from "./db/index.js";
import { rankingResponses } from "./rerank/index.js";

interface SearchResult {
  id: number;
  text: string;
  _distance?: number;
  chapter: string;
}

const OLLAMA_URL = "http://localhost:11434";
const LLM_MODEL = "mistral:7b-instruct";
const COLLECTION_NAME = "documents";
const TEXTCHUNKWIDE = 2;

async function askQuestion(question: string): Promise<void> {
  const db = await connectToVectorDB();
  const table = await db.openTable(COLLECTION_NAME);
  const queryEmbedding: number[] = await generateEmbedding(question);

  // In 'vectordb' (JS), .execute() returns an Array directly
  const results: SearchResult[] = await table
    .search(queryEmbedding)
    .limit(3)
    .execute();

  if (results.length === 0) {
    console.log("No relevant information found.");
    return;
  }

  const contextChunks: Set<string> = new Set<string>();

  for (const res of results) {
    const idx: number = Number(res.id);
    if (isNaN(idx)) {
      contextChunks.add(res.text);
      continue;
    }
    const totalRows: number = await table.countRows();
    const start: number = Math.max(0, idx - TEXTCHUNKWIDE);
    const end: number = Math.min(totalRows, idx + TEXTCHUNKWIDE);

    const neighbors: SearchResult[] = await table
      .filter(`id >= ${start} AND id <= ${end} AND chapter = '${res.chapter}'`) //only same chapter context
      .execute();
    
    neighbors.sort((a, b) => a.id - b.id);

    const enrichedText: string = `[CONTEXTO: Información extraída del ${res.chapter}] y el contenido es:\n` + 
      neighbors
        .map(n => n.text)
        .join(" ");

    if (enrichedText.trim().length > 0) {
      contextChunks.add(enrichedText);
    }
  }
  const reRankedContextChunks: string[] = await rankingResponses(question, Array.from(contextChunks));
  if (reRankedContextChunks.length === 0) {
    console.log("No relevant information found.");
    return;
  }
  const finalContext: string = Array.from(reRankedContextChunks).join("\n---\n");
  console.log(finalContext.length)
  try {
    const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
      model: LLM_MODEL,
      prompt: `Usa el CONTEXTO para responder la PREGUNTA, no te inventes nada ni hagas suposiciones, responde
      con la información contenida en el CONTEXTO!.\n\nCONTEXTO:\n${finalContext}\n\nPREGUNTA: ${question}`,
      stream: false
    });

    console.log("\nRESPUESTA del LLM local:\n" + response.data.response);
  } catch (error: any) {
    console.error("Error en Ollama:", error.message);
  }
}

async function main(): Promise<void> {
  try {
    const archivoALeer = "/home/msi/Desktop/AI/RAG/rag-local/texto.txt"; 
    console.log(process.env.COHERE_API_KEY);
    if (!fs.existsSync(archivoALeer)) {
      console.error(`El archivo ${archivoALeer} no existe.`);
      return;
    }

    console.log("⏳ Indexando documento...");
    await indexDocument(archivoALeer);
    
    const rl = readline.createInterface({ input, output });
    console.log("\nSistema listo. Escribe 'salir' para terminar.");

    while (true) {
      const pregunta = await rl.question("\nHaz tu pregunta: ");
      if (["salir", "exit", "quit"].includes(pregunta.toLowerCase())) break;
      if (!pregunta.trim()) continue;

      await askQuestion(pregunta);
    }
    rl.close();
  } catch (error) {
    console.error("Error crítico:", error);
  }
}

main();


//Mejoras:
//Ajusta modelo, temperatura y ks
//Ajusta el contexto
//Ajusta el ranking, para que si no devuelve respuestas ajustadas, no haya respuesta por parte del modelo => ahorro de costes.


