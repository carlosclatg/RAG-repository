import axios from 'axios';

const OLLAMA_URL = 'http://localhost:11434';
const LLM_MODEL_RESPONSE = 'mistral:7b-instruct';
const LLM_MODEL_SEARCH_DECISION = 'deepseek-r1:1.5b';
export const SEMANTINC_MODE = 'semantico';
export const HYBRID_MODE = 'hibrido';
export type GeneratorResult =
	| { success: true; data: string }
	| { success: false; error: string };

export async function generateResponse(
	query: string,
	finalContext: string,
): Promise<GeneratorResult> {
	try {
		const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
			model: LLM_MODEL_RESPONSE,
			prompt: `Usa el CONTEXTO para responder la PREGUNTA, no te inventes nada ni hagas suposiciones, responde
            con la información contenida en el CONTEXTO!.\n\nCONTEXTO:\n${finalContext}\n\nPREGUNTA: ${query}`,
			stream: false,
		});

		return { success: true, data: response.data.response };
	} catch (error: any) {
		// Diferenciamos tipos de errores
		let message = 'Error desconocido';

		if (error.code === 'ECONNREFUSED') {
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

export async function selectRAGMode(prompt: string): Promise<string> {
	const routerPrompt = (prompt: string) => `
        ESTRICTA RESPUESTA EN JSON.
        Eres un experto en extracción de entidades. Tu tarea es decidir si la pregunta requiere una búsqueda SEMÁNTICA o HÍBRIDA.
        
        REGLAS:
        1. Si la pregunta menciona NOMBRES PROPIOS (Lyra, Kael, Silas, Aeridor), LUGARES o OBJETOS específicos, usa mode ${HYBRID_MODE}.
        2. Si la pregunta es sobre conceptos, sentimientos o temas generales, usa mode ${SEMANTINC_MODE}.
        3. En "filter", pon solo el nombre propio encontrado. Si no hay, pon null.
        4. Omite cualquier otro texto, limítate a dar respuesta en formato JSON.

        FORMATO DE SALIDA:
        {"mode": "hibrido" | "semantico", "filter": string | null}

        PREGUNTA: "${prompt}"
        JSON:
        `;

	try {
		const response = await axios.post(`${OLLAMA_URL}/api/generate`, {
			model: LLM_MODEL_RESPONSE,
			prompt: routerPrompt(prompt),
			stream: false,
			options: {
				temperature: 0,
			},
		});
		return await response.data.response;
	} catch (error: any) {
		console.error('Error en Ollama:', error.message);
		return Promise.reject(error);
	}
}
