import axios from 'axios';
import { OLLAMA_URL, LLM_MODEL } from '../config/index.js';

export const SEMANTIC_MODE = 'semantico';
export const HYBRID_MODE = 'hibrido';
export type GeneratorResult =
	| { success: true; data: string }
	| { success: false; error: string };

export async function generateResponse(
	query: string,
	finalContext: string,
): Promise<GeneratorResult> {
	try {
		process.stdout.write('\nRespuesta: ');

		const response = await axios.post(
			`${OLLAMA_URL}/api/generate`,
			{
				model: LLM_MODEL,
				prompt: `Usa el CONTEXTO para responder la PREGUNTA. No te inventes nada ni hagas suposiciones; responde únicamente con la información contenida en el CONTEXTO.\n\nCONTEXTO:\n${finalContext}\n\nPREGUNTA: ${query}`,
				stream: true,
				options: { temperature: 0 },
			},
			{ responseType: 'stream' },
		);

		return await new Promise((resolve) => {
			let fullResponse = '';

			response.data.on('data', (chunk: Buffer) => {
				const lines = chunk.toString().split('\n').filter(Boolean);
				for (const line of lines) {
					try {
						const parsed = JSON.parse(line);
						if (parsed.response) {
							process.stdout.write(parsed.response);
							fullResponse += parsed.response;
						}
					} catch {
						// partial chunk, ignore
					}
				}
			});

			response.data.on('end', () => {
				process.stdout.write('\n');
				resolve({ success: true, data: fullResponse });
			});

			response.data.on('error', (err: Error) => {
				resolve({ success: false, error: err.message });
			});
		});
	} catch (error: unknown) {
		const axiosError = error as {
			code?: string;
			response?: { status: number };
			message?: string;
		};
		let message = 'Unknown error';
		if (axiosError.code === 'ECONNREFUSED') {
			message = 'Could not connect to Ollama. Is the server running?';
		} else if (axiosError.response) {
			message = `Ollama responded with error: ${axiosError.response.status}`;
		} else {
			message = axiosError.message ?? message;
		}
		console.error(`[LLM_ERROR]: ${message}`);
		return { success: false, error: message };
	}
}

export async function selectRAGMode(prompt: string): Promise<string> {
	const routerPrompt = (p: string) => `
        STRICT JSON RESPONSE ONLY.
        You are an entity extraction expert. Decide if the question requires SEMANTIC or HYBRID search.
        
        RULES:
        1. If the question mentions PROPER NOUNS (names, places, specific objects), use mode ${HYBRID_MODE}.
        2. If the question is about concepts, feelings, or general topics, use mode ${SEMANTIC_MODE}.
        3. In "filter", put only the proper noun found. If none, put null.
        4. Respond ONLY with the JSON object, no other text.

        OUTPUT FORMAT:
        {"mode": "hibrido" | "semantico", "filter": string | null}

        QUESTION: "${p}"
        JSON:
        `;

	try {
		const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
			model: LLM_MODEL,
			prompt: routerPrompt(prompt),
			stream: false,
			options: { temperature: 0 },
		});
		return response.data.response;
	} catch (error: unknown) {
		const err = error as { message?: string };
		console.error('[LLM_ERROR] selectRAGMode:', err.message);
		throw error;
	}
}
