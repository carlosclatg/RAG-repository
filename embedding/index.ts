//import pdf from "pdf-parse-fork";
import { chunkText, recursiveChunkingBySentences } from '../chunk/index.ts';
import * as fs from 'fs';
import { connectToVectorDB } from '../db/index.ts';
import * as lancedb from 'vectordb';
import axios, { AxiosResponse } from 'axios';
import { Index } from '@lancedb/lancedb';

interface EmbeddingResponse {
	embedding: number[];
}

const EMBEDDING_MODEL: string = 'nomic-embed-text';
const OLLAMA_URL: string = 'http://localhost:11434';

export async function generateEmbedding(text: string): Promise<number[]> {
	try {
		const response: AxiosResponse<EmbeddingResponse> = await axios.post(
			`${OLLAMA_URL}/api/embeddings`,
			{
				model: EMBEDDING_MODEL,
				prompt: text,
			},
		);
		return response.data.embedding;
	} catch (error: any) {
		console.error('Error while generating embedding.');
		console.error(error);
		throw error;
	}
}
