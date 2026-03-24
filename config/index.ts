export const OLLAMA_URL: string =
	process.env.OLLAMA_URL ?? 'http://localhost:11434';

export const LLM_MODEL: string = process.env.LLM_MODEL ?? 'mistral:7b-instruct';

export const EMBEDDING_MODEL: string =
	process.env.EMBEDDING_MODEL ?? 'nomic-embed-text';

export const DB_PATH: string = process.env.DB_PATH ?? './data';

export const COHERE_API_KEY: string | undefined = process.env.COHERE_API_KEY;

export const DOCUMENT_PATH: string = process.env.DOCUMENT_PATH ?? './texto.txt';

export const RERANK_THRESHOLD: number = Number(
	process.env.RERANK_THRESHOLD ?? 0.8,
);

export const QUERY_LIMIT: number = Number(process.env.QUERY_LIMIT ?? 20);

export const TEXT_CHUNK_WIDE: number = Number(process.env.TEXT_CHUNK_WIDE ?? 2);
