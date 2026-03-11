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

interface SearchResult {
	id: number;
	text: string;
	_distance?: number;
	chapter: string;
}

const COLLECTION_NAME = 'documents';
const TEXTCHUNKWIDE = 2;

async function askQuestion(question: string): Promise<void> {
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);
	const queryEmbedding: number[] = await generateEmbedding(question);

	const modeSelection = await selectRAGMode(question);
	const { mode, filter } = JSON.parse(modeSelection);

	let results: SearchResult[] = [];
	if (mode === SEMANTINC_MODE) {
		results = await table
			.search(queryEmbedding) // Busca por significado (Vector)
			.limit(3)
			.toArray();
	}
	if (mode === HYBRID_MODE) {
		results = await table.search(filter).limit(20).toArray(); //Busca por FTS
	}

	if (results.length === 0) {
		console.log('No relevant information found.');
		return;
	}

	const contextChunks: Set<string> = new Set<string>();
	const alreadyProcessedResult = new Set<number>(); //to avoid process the repeated results
	for (const res of results) {
		const idx: number = Number(res.id);
		if (isNaN(idx)) {
			contextChunks.add(res.text);
			continue;
		}
		if (alreadyProcessedResult.has(idx)) {
			continue;
		}
		alreadyProcessedResult.add(idx);
		console.log(`[CONTEXTO: Información extraída del ${idx}]`);
		const totalRows: number = await table.countRows();
		const start: number = Math.max(0, idx - TEXTCHUNKWIDE);
		const end: number = Math.min(totalRows, idx + TEXTCHUNKWIDE);
		const neighbors: SearchResult[] = await table
			.query()
			.where(
				`id >= ${start} AND id <= ${end} AND chapter = '${res.chapter}'`,
			)
			.toArray();

		neighbors.sort((a, b) => a.id - b.id);
		const enrichedText: string =
			`[CONTEXTO: Información extraída del ${res.chapter}] y el contenido es:\n` +
			neighbors.map((n) => n.text).join(' ');

		if (enrichedText.trim().length > 0) {
			contextChunks.add(enrichedText);
		}
	}
	const reRankedContextChunks: string[] = await rankingResponses(
		question,
		Array.from(contextChunks),
	);
	if (reRankedContextChunks.length === 0) {
		console.log('No relevant information found.');
		return;
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

//Mejoras:
//Ajusta modelo, temperatura y ks
//Ajusta el contexto
//Ajusta el ranking, para que si no devuelve respuestas ajustadas, no haya respuesta por parte del modelo => ahorro de costes.
