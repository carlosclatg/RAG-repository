import * as fs from 'fs';
import * as readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import { generateEmbedding } from './embedding/index.js';
import { connectToVectorDB, indexDocument } from './db/index.js';
import { rankingResponses } from './rerank/index.js';
import {
	generateResponse,
	GeneratorResult,
	HYBRID_MODE,
	selectRAGMode,
	SEMANTIC_MODE,
} from './llm/index.js';
import { Table } from '@lancedb/lancedb';
import { DOCUMENT_PATH, QUERY_LIMIT, TEXT_CHUNK_WIDE } from './config/index.js';

interface SearchResult {
	id: number;
	text: string;
	_distance?: number;
	chapter: string;
}

const COLLECTION_NAME = 'documents';
const MAX_CHAPTERS_TO_TRY = 10;

async function askQuestion(question: string): Promise<void> {
	// usedChapters is scoped per question so state never leaks between queries
	const usedChapters: string[] = [];

	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);
	const queryEmbedding: number[] = await generateEmbedding(question);
	const modeSelection = await selectRAGMode(question);
	const { mode, filter } = JSON.parse(modeSelection);
	let reRankedContextChunks: string[] = [];
	let results: SearchResult[] = [];

	if (mode === SEMANTIC_MODE) {
		results = (await table
			.search(queryEmbedding)
			.limit(QUERY_LIMIT)
			.toArray()) as SearchResult[];
		reRankedContextChunks = await semanticModeSearch(
			results,
			table,
			question,
		);
	}

	if (mode === HYBRID_MODE) {
		results = (await table
			.search(filter)
			.limit(QUERY_LIMIT)
			.toArray()) as SearchResult[];
		reRankedContextChunks = await hybridModeSearch(
			results,
			table,
			question,
			usedChapters,
		);
	}

	if (reRankedContextChunks.length === 0) {
		console.log('\nNo relevant information found for your question.');
		return;
	}

	const finalContext: string = reRankedContextChunks.join('\n---\n');
	const result: GeneratorResult = await generateResponse(
		question,
		finalContext,
	);

	if (!result.success) {
		console.error('\nError:', result.error);
	}
}

async function hybridModeSearch(
	results: SearchResult[],
	table: Table,
	question: string,
	usedChapters: string[],
): Promise<string[]> {
	let reRankedContextChunks: string[] = [];

	while (
		reRankedContextChunks.length === 0 &&
		usedChapters.length < MAX_CHAPTERS_TO_TRY
	) {
		if (results.length === 0) break;

		const currentChapterToProcess =
			results.find(
				(res) => res.chapter && !usedChapters.includes(res.chapter),
			)?.chapter ?? null;

		if (!currentChapterToProcess) {
			console.log('No more new chapters to process in current results.');
			break;
		}

		usedChapters.push(currentChapterToProcess);
		const chapterText = await getSingleChapterContent(
			table,
			currentChapterToProcess,
		);

		if (chapterText.trim().length > 0) {
			const contextPayload = [
				`[CHAPTER: ${currentChapterToProcess}]\n${chapterText}`,
			];
			console.log(`Trying chapter: ${currentChapterToProcess}...`);
			reRankedContextChunks = await rankingResponses(
				question,
				contextPayload,
			);

			if (reRankedContextChunks.length === 0) {
				console.log(
					`Chapter "${currentChapterToProcess}" had no relevant info. Trying next...`,
				);
			}
		}
	}

	return reRankedContextChunks;
}

async function semanticModeSearch(
	results: SearchResult[],
	table: Table,
	question: string,
): Promise<string[]> {
	if (results.length === 0) {
		return [];
	}
	const contextChunks = await getContextFromNeighbors(
		table,
		results,
		TEXT_CHUNK_WIDE,
	);
	return rankingResponses(question, Array.from(contextChunks));
}

async function main(): Promise<void> {
	const documentPath = DOCUMENT_PATH;

	if (!fs.existsSync(documentPath)) {
		console.error(`Document not found: ${documentPath}`);
		console.error(
			'Set the DOCUMENT_PATH environment variable to the correct path.',
		);
		process.exit(1);
	}

	const db = await connectToVectorDB();
	const tableNames = await db.tableNames();
	const tableExists = tableNames.includes(COLLECTION_NAME);
	let tableHasRows = false;

	if (tableExists) {
		const table = await db.openTable(COLLECTION_NAME);
		tableHasRows = (await table.countRows()) > 0;
	}

	if (!tableHasRows) {
		console.log('⏳ Indexing document...');
		await indexDocument(documentPath);
		console.log('✅ Document indexed.');
	} else {
		console.log('✅ Index already exists, skipping re-indexing.');
	}

	const rl = readline.createInterface({ input, output });

	const shutdown = () => {
		console.log('\nGoodbye!');
		rl.close();
		process.exit(0);
	};
	process.on('SIGINT', shutdown);
	process.on('SIGTERM', shutdown);

	console.log("\nSystem ready. Type 'exit' to quit.");

	try {
		while (true) {
			const question = await rl.question('\nYour question: ');
			if (['salir', 'exit', 'quit'].includes(question.toLowerCase()))
				break;
			if (!question.trim()) continue;

			try {
				await askQuestion(question);
			} catch (error) {
				console.error('Error processing question:', error);
			}
		}
	} finally {
		rl.close();
	}
}

main().catch((error) => {
	console.error('Fatal error:', error);
	process.exit(1);
});

/**
 * Fetches all chunks belonging to a chapter, sorted by id, and joins them.
 * Uses parameterized-style escaping to prevent injection via chapter names.
 */
async function getSingleChapterContent(
	table: Table,
	chapterName: string,
): Promise<string> {
	// Escape single quotes to prevent injection through chapter names
	const safeChapterName = chapterName.replace(/'/g, "''");
	const chapterChunks = (await table
		.query()
		.where(`chapter = '${safeChapterName}'`)
		.toArray()) as SearchResult[];

	return chapterChunks
		.sort((a, b) => Number(a.id) - Number(b.id))
		.map((c) => c.text)
		.join(' ');
}

/**
 * Returns context chunks around each search result, expanding by textChunkWide
 * neighbours on each side within the same chapter.
 */
async function getContextFromNeighbors(
	table: Table,
	results: SearchResult[],
	textChunkWide: number,
): Promise<Set<string>> {
	const contextChunks = new Set<string>();
	const alreadyProcessed = new Set<number>();
	const totalRows: number = await table.countRows();

	for (const res of results) {
		const idx = Number(res.id);

		if (isNaN(idx)) {
			contextChunks.add(res.text);
			continue;
		}
		if (alreadyProcessed.has(idx)) continue;
		alreadyProcessed.add(idx);

		const start = Math.max(0, idx - textChunkWide);
		const end = Math.min(totalRows - 1, idx + textChunkWide);
		const safeChapter = res.chapter.replace(/'/g, "''");

		const neighbors = (await table
			.query()
			.where(
				`id >= ${start} AND id <= ${end} AND chapter = '${safeChapter}'`,
			)
			.toArray()) as SearchResult[];

		neighbors.sort((a, b) => a.id - b.id);

		const enrichedText =
			`[CONTEXT: Chapter ${res.chapter}]\n` +
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
