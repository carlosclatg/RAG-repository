import * as fs from 'fs';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import axios from 'axios';
import { generateEmbedding } from './embedding/index.js';
import { connectToVectorDB, indexDocument } from './db/index.js';
import { rankingResponses } from './rerank/index.js';
import {
	generateResponse,
	GeneratorResult,
	HYBRID_MODE,
	selectRAGMode,
	SEMANTINC_MODE,
} from './llm/index.js';
import { Table } from '@lancedb/lancedb';

interface SearchResult {
	id: number;
	text: string;
	_distance?: number;
	chapter: string;
}

const COLLECTION_NAME = 'documents';
const TEXTCHUNKWIDE = 2;
let usedChapters: string[] = [];
const QUERY_LIMIT = 20;
//MEJORA:
//1. Obtener resultados de busqueda
//2. Obtener los n mejores capítulos
//3. Pasarlos al ranker, sino tienen buena coincidencia, pasar a los siguientes capítulos.
//4. Si la respuesta es buena, en ese caso pasarlo al LLM para la respuesta definitiva.
//5. Si no hubiera una respuesta adecuada, en tal caso, se debería responder por parte del LLM, que no hay ninguna respuesta adecuada.
async function askQuestion(question: string): Promise<void> {
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);
	const queryEmbedding: number[] = await generateEmbedding(question);
	const semantincModeEnabled = false;
	const modeSelection = await selectRAGMode(question);
	const { mode, filter } = JSON.parse(modeSelection);
	let reRankedContextChunks: string[] = [];
	let results: SearchResult[] = [];
		if (semantincModeEnabled || mode === SEMANTINC_MODE) {
			results = await table
				.search(queryEmbedding) // Busca por significado (Vector)
				.limit(QUERY_LIMIT)
				.toArray();
			//Estrategia de mirar el contexto del vector, es decir  vector anterior y posterior.
			reRankedContextChunks = await semanticModeSearch(reRankedContextChunks, results, table, question);
		}
		if (mode === HYBRID_MODE) {
			results = await table
				.search(filter)
				.limit(QUERY_LIMIT)
				.toArray(); //Busca por FTS
			//Estrategia de mirar el contexto del capítulo entero dado un vector con un capítulo.
			reRankedContextChunks = await hybridModeSearch(reRankedContextChunks, results, table, question);
		}
	
	const finalContext: string = Array.from(reRankedContextChunks).join(
		'\n---\n',
	);
	const result: GeneratorResult = await generateResponse(
		question,
		finalContext,
	);

	if (result.success) {
		console.log('Respuesta:', result.data);
	} else {
		// Aquí decides qué mostrar al usuario final
		console.error('Hubo un problema:', result.error);
	}
}


async function hybridModeSearch(reRankedContextChunks: string[], results: SearchResult[], table: Table, question: string) {
	while (reRankedContextChunks.length === 0 && usedChapters.length < 10) {
		if (results.length === 0) break;

		let currentChapterToProcess: string | null = null;

		// 1. Buscamos el primer resultado cuyo capítulo NO hayamos utilizado
		for (const res of results) {
			if (res.chapter && !usedChapters.includes(res.chapter)) {
				currentChapterToProcess = res.chapter;
				break;
			}
		}

		// Si no quedan capítulos nuevos en la lista de resultados, salimos del while
		if (!currentChapterToProcess) {
			console.log('No quedan más capítulos nuevos por procesar en los resultados actuales.');
			break;
		}

		// 2. Marcamos como usado y obtenemos el contenido
		usedChapters.push(currentChapterToProcess);
		const chapterText = await getSingleChapterContent(table, currentChapterToProcess);

		if (chapterText.trim().length > 0) {
			const contextPayload = [`[CAPÍTULO: ${currentChapterToProcess}]\n${chapterText}`];

			// 3. Pasamos al reranker inmediatamente
			console.log(`Probando suerte con el capítulo: ${currentChapterToProcess}...`);
			reRankedContextChunks = await rankingResponses(question, contextPayload);

			// Si el reranker devuelve algo, el 'while' terminará automáticamente 
			// porque reRankedContextChunks.length ya no será 0.
			if (reRankedContextChunks.length === 0) {
				console.log(`El capítulo ${currentChapterToProcess} no aportó info útil. Buscando el siguiente...`);
			}
		}
	}
	return reRankedContextChunks;
}

async function semanticModeSearch(reRankedContextChunks: string[], results: SearchResult[], table: Table, question: string) {
	while (reRankedContextChunks.length === 0) {
		if (results.length === 0) {
			console.log('No relevant information found.');
			break;
		}
		const contextChunks: Set<string> = await getContextFromNeighbors(table, results, TEXTCHUNKWIDE);
		reRankedContextChunks = await rankingResponses(
			question,
			Array.from(contextChunks)
		);
		if (reRankedContextChunks.length === 0) {
			console.log('No relevant information found.');
		}
	}
	return reRankedContextChunks;
}

async function main(): Promise<void> {
	try {
		const archivoALeer = '/home/msi/Desktop/AI/RAG/rag-local/texto.txt';
		if (!fs.existsSync(archivoALeer)) {
			console.error(`El archivo ${archivoALeer} no existe.`);
			return;
		}

		console.log('⏳ Indexando documento...');
		await indexDocument(archivoALeer);

		const rl = readline.createInterface({ input, output });
		console.log("\nSistema listo. Escribe 'salir' para terminar.");

		while (true) {
			const pregunta = await rl.question('\nHaz tu pregunta: ');
			if (['salir', 'exit', 'quit'].includes(pregunta.toLowerCase()))
				break;
			if (!pregunta.trim()) continue;

			await askQuestion(pregunta);
		}
		rl.close();
	} catch (error) {
		console.error('Error crítico:', error);
	}
}

main();

async function getSingleChapterContent(table: any, chapterName: string): Promise<string> {
    const chapterChunks = await table
        .query()
        .where(`chapter = '${chapterName}'`)
        .toArray();

    // Ordenar por ID numérico y unir texto
    return chapterChunks
        .sort((a: any, b: any) => Number(a.id) - Number(b.id))
        .map((c: any) => c.text)
        .join(' ');
}


/**
 * 
 * @param table 
 * @param results 
 * @param textChunkWide 
 * @returns returns the context chunks around the results [vector - textChunkWide, ..., vector + textChunkWide]
 */
async function getContextFromNeighbors(
    table: any, 
    results: SearchResult[], 
    textChunkWide: number
): Promise<Set<string>> {
    const contextChunks: Set<string> = new Set<string>();
    const alreadyProcessedResult = new Set<number>();

    for (const res of results) {
        const idx: number = Number(res.id);
        
        // Validaciones básicas
        if (isNaN(idx)) {
            contextChunks.add(res.text);
            continue;
        }
        if (alreadyProcessedResult.has(idx)) continue;

        alreadyProcessedResult.add(idx);
        
        // Lógica de vecindad
        const totalRows: number = await table.countRows();
        const start: number = Math.max(0, idx - textChunkWide);
        const end: number = Math.min(totalRows, idx + textChunkWide);

        const neighbors: SearchResult[] = await table
            .query()
            .where(`id >= ${start} AND id <= ${end} AND chapter = '${res.chapter}'`)
            .toArray();

        neighbors.sort((a, b) => a.id - b.id);
        
        const enrichedText = `[CONTEXTO: Capítulo ${res.chapter}] Contenido:\n` + 
                             neighbors.map((n) => n.text).join(' ');

        if (enrichedText.trim().length > 0) {
            contextChunks.add(enrichedText);
        }
    }
    return contextChunks;
}
//Mejoras:
//Ajusta modelo, temperatura y ks
//Ajusta el contexto
//Ajusta el ranking, para que si no devuelve respuestas ajustadas, no haya respuesta por parte del modelo => ahorro de costes.
