import axios, { AxiosResponse } from 'axios';
import { OLLAMA_URL, EMBEDDING_MODEL } from '../config/index.js';

interface EmbeddingResponse {
	embedding: number[];
}

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
	} catch (error: unknown) {
		console.error('Error while generating embedding:', error);
		throw error;
	}
}
