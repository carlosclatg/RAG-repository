import * as lancedb from '@lancedb/lancedb';
import { recursiveChunkingBySentences } from '../chunk/index.js';
import * as fs from 'fs';
import { Index } from '@lancedb/lancedb';
import { generateEmbedding } from '../embedding/index.js';
import { DB_PATH, DOCUMENT_PATH } from '../config/index.js';

let instance: lancedb.Connection; //singleton instance
const COLLECTION_NAME: string = 'documents';

interface VectorRow {
	id: number;
	vector: number[];
	text: string;
	chapter: string;
	[key: string]: unknown;
}
export async function connectToVectorDB(): Promise<lancedb.Connection> {
	if (!instance) {
		instance = await lancedb.connect(DB_PATH);
	}
	return Promise.resolve(instance);
}

export async function openTable(
	collectionName: string,
): Promise<lancedb.Table> {
	return await instance.openTable(collectionName);
}

export async function indexDocument(path: string): Promise<void> {
	console.log('Extrayendo contenido...');
	const fullText: string = fs.readFileSync(path, 'utf8');

	// 1. Split by chapters and detect the chapter strings. "Capítulo X: Título"
	const chapterRegex = /(Capítulo\s+\d+:?\s*[^\n]+)/gi;
	const sections = fullText.split(chapterRegex);

	const rows: VectorRow[] = [];
	let globalChunkIndex = 0;
	let currentChapterTitle = 'Introducción / Prólogo';

	console.log('Procesando secciones y generando chunks enriquecidos...');

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

			if (embedding?.length > 0) {
				rows.push({
					id: globalChunkIndex++,
					vector: Array.from(embedding),
					text: subChunk,
					chapter: currentChapterTitle,
				});
			}
		}
	}

	const db = await connectToVectorDB();
	const tableNames = await db.tableNames();

	if (tableNames.includes(COLLECTION_NAME)) {
		const table = await db.openTable(COLLECTION_NAME);
		await table.add(rows);
		await table.createIndex('text', {
			config: Index.fts(),
			replace: true,
		});
		console.log(await table.listIndices());
	} else {
		const table = await db.createTable(COLLECTION_NAME, rows);
		await table.createIndex('text', {
			config: Index.fts(),
			replace: true,
		});
		console.log(await table.listIndices());
	}
	console.log('Documento indexado.');
}

export async function initDBAndChunking() {
	let documentPath = DOCUMENT_PATH;
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
}
