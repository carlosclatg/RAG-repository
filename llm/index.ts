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
		const prompt = `
				Instrucciones: Eres un asistente experto. Responde EXCLUSIVAMENTE usando el CONTEXTO proporcionado. Si la respuesta no está en el contexto, di que no lo sabes.
				CONTEXTO:
				${finalContext}
				PREGUNTA:
				${query}
				`;
		const cleanPrompt = preparePromptForOllama(prompt);
		const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
			model: LLM_MODEL,
			prompt: cleanPrompt,
			stream: false,
			options: {
				temperature: 0,
				num_ctx: 32768,
				repeat_penalty: 1.1,
				stop: ['###', 'Instrucciones:'],
			},
		});

		return { success: true, data: response.data.response };
	} catch (error: unknown) {
		// Diferenciamos tipos de errores
		let message = 'Error desconocido';

		if (error instanceof Error && error.code === 'ECONNREFUSED') {
			message =
				'No se pudo conectar con Ollama. ¿Está el servidor encendido?';
		} else if (error.response) {
			message = `Ollama respondió con error: ${error.response.status}`;
		} else {
			message = error.message;
		}

		console.error(`[LLM_ERROR]: ${message}`);
		return { success: false, error: message };
	}
}

function preparePromptForOllama(text: string): string {
	return text
		.replace(/\r?\n|\r/g, ' ') // Cambia cualquier tipo de salto de línea por un espacio
		.replace(/\t/g, ' ') // Cambia tabuladores por espacios
		.replace(/\s+/g, ' ') // Si hay 2 o más espacios juntos, los deja en 1 solo
		.trim(); // Quita espacios al principio y al final
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
