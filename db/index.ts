import * as lancedb from '@lancedb/lancedb';
//import pdf from "pdf-parse-fork";
import { chunkText, recursiveChunkingBySentences } from '../chunk/index.ts';
import * as fs from 'fs';
import axios, { AxiosResponse } from 'axios';
import { Index } from '@lancedb/lancedb';
import { generateEmbedding } from '../embedding/index.ts';
const DB_PATH: string = './data';

let instance: lancedb.Connection; //singleton instance
const COLLECTION_NAME: string = 'documents';

interface VectorRow {
	id: number;
	vector: number[];
	text: string;
	chapter: string;
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

			if (embedding?.length === 768) {
				rows.push({
					id: globalChunkIndex++,
					vector: Array.from(embedding),
					text: subChunk,
					chapter: currentChapterTitle, //add title of chapter as metadata
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
			config: Index.fts(), // Indice de Full Text Search con inverted index.
			replace: true,
		});
		console.log(table.listIndices());
	} else {
		const table = await db.createTable(COLLECTION_NAME, rows);
		await table.createIndex('text', {
			config: Index.fts(), // Indice de Full Text Search con inverted index.
			replace: true,
		});
		console.log(await table.listIndices());
	}
	console.log('Documento indexado.');
}
