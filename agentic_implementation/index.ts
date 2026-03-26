/**
 * RAG Agent con Bucle de Reintento (Re-search Loop)
 * ===============================================
 * [__start__]
 * │
 * [routeQuestion] ─── decides: "semantic" | "hybrid" | "no_retrieval"
 * │
 * ┌───────┴──────────────┬──────────────┐
 * │                      │              │
 * │               (Si no hay filtro)    │
 * │                      │              │
 * [semanticSearch] <─── [hybridSearch]    │
 * │        ^             │              │
 * │        │             │              │
 * └───────┬┴─────────────┘              │
 * │                             │
 * [gradeDocuments] <──────────────────┘
 * │  decides: "relevant" | "not_relevant"
 * │
 * ┌─────┴───────┬──────────────────┐
 * │             │                  │
 * (relevant)  (not_relevant)     (no_retrieval o
 * │        e intentos          intentos agotados)
 * │        restantes)              │
 * │             │                  │
 * V             V                  V
 [generateAnswer]   (Volver a buscar)   [noAnswer]
 * │                                │
 * [__end__]                        [__end__]
 */

import { Annotation, StateGraph, END, START } from '@langchain/langgraph';
import { generateEmbedding } from '../embedding/index.ts';
import { connectToVectorDB, initDBAndChunking } from '../db/index.ts';
import { generateResponse, selectRAGMode, HYBRID_MODE } from '../llm/index.ts';
import { rankingResponses } from '../rerank_service/index.ts';
import { QUERY_LIMIT } from '../config/index.js';
import { SqliteSaver } from '@langchain/langgraph-checkpoint-sqlite';
import { randomUUID } from 'crypto';

// ─────────────────────────────────────────────────────────────────────────────
// 1. STATE DEFINITION
// ─────────────────────────────────────────────────────────────────────────────

interface DocumentChunk {
	id: number;
	text: string;
	chapter: string;
	_distance?: number;
}

const AgentState = Annotation.Root({
	question: Annotation<string>(),
	mode: Annotation<string>(),
	filter: Annotation<string | null>(),
	documents: Annotation<DocumentChunk[]>({
		value: (_prev, next) => next,
		default: () => [],
	}),
	relevantChunks: Annotation<string[]>({
		value: (_prev, next) => next,
		default: () => [],
	}),
	answer: Annotation<string>(),
	/** * NUEVO: Registro de métodos ya probados para evitar bucles infinitos
	 * y permitir la lógica de reintento.
	 */
	triedModes: Annotation<string[]>({
		value: (prev, next) => [...prev, ...next],
		default: () => [],
	}),
});

type State = typeof AgentState.State;
const COLLECTION_NAME = 'documents';

// ─────────────────────────────────────────────────────────────────────────────
// 2. NODES
// ─────────────────────────────────────────────────────────────────────────────

async function routeQuestion(state: State) {
	console.log('\n[routeQuestion] Decidiendo estrategia inicial...');
	try {
		const raw = await selectRAGMode(state.question);
		const { mode, filter } = JSON.parse(raw);
		console.log(`[routeQuestion] → modo sugerido: ${mode}`);
		return { mode, filter };
	} catch {
		return { mode: 'semantico', filter: null };
	}
}

async function semanticSearch(state: State) {
	console.log('\n[semanticSearch] Ejecutando búsqueda vectorial...');
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);
	const embedding = await generateEmbedding(state.question);

	const results = (await table
		.search(embedding)
		.limit(QUERY_LIMIT)
		.toArray()) as DocumentChunk[];

	console.log(`[semanticSearch] Encontrados ${results.length} chunks.`);
	return {
		documents: results,
		triedModes: ['semantico'],
	};
}

async function hybridSearch(state: State) {
	console.log('\n[hybridSearch] Ejecutando búsqueda FTS (Híbrida)...');
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);

	if (!state.filter) {
		console.warn(
			'[hybridSearch] Sin filtro FTS, redirigiendo a semántico.',
		);
		return semanticSearch(state);
	}

	const results = (await table
		.search(state.filter)
		.limit(QUERY_LIMIT)
		.toArray()) as DocumentChunk[];

	console.log(`[hybridSearch] Encontrados ${results.length} chunks.`);
	return {
		documents: results,
		triedModes: [HYBRID_MODE],
	};
}

async function gradeDocuments(state: State) {
	console.log('\n[gradeDocuments] Re-rankeando y evaluando relevancia...');
	if (state.documents.length === 0) return { relevantChunks: [] };

	const rawTexts = state.documents.map(
		(d) => `[Chapter: ${d.chapter}]\n${d.text}`,
	);

	const relevantChunks = await rankingResponses(state.question, rawTexts);
	console.log(
		`[gradeDocuments] ${relevantChunks.length} chunks pasaron el filtro.`,
	);
	return { relevantChunks };
}

async function generateAnswer(state: State) {
	console.log('\n[generateAnswer] Generando respuesta final...');
	const context = state.relevantChunks.join('\n---\n');
	const result = await generateResponse(state.question, context);

	return { answer: result.success ? result.data : `Error: ${result.error}` };
}

function noAnswer(state: State) {
	console.log('\n[noAnswer] Agotadas todas las vías de búsqueda.');
	return {
		answer: `Lo siento, tras buscar por varios métodos, no encontré información relevante para: "${state.question}"`,
	};
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. CONDITIONAL EDGES (LOGIC)
// ─────────────────────────────────────────────────────────────────────────────

function decideSearchMode(state: State): string {
	if (state.mode === HYBRID_MODE) return 'hybridSearch';
	if (state.mode === 'no_retrieval') return 'noAnswer';
	return 'semanticSearch';
}

/**
 * LÓGICA DE REINTENTO:
 * Si no hay chunks relevantes, comprobamos qué modo NO hemos probado aún.
 */
function decideAfterGrading(state: State): string {
	if (state.relevantChunks.length > 0) {
		return 'generateAnswer';
	}

	// Si falló el híbrido pero no hemos probado el semántico
	if (
		state.triedModes.includes(HYBRID_MODE) &&
		!state.triedModes.includes('semantico')
	) {
		console.log(
			'🔄 RE-INTENTO: El modo híbrido falló. Probando búsqueda semántica...',
		);
		return 'semanticSearch';
	}

	// Si falló el semántico pero el router nos dio un filtro y no hemos probado híbrido
	if (
		state.triedModes.includes('semantico') &&
		!state.triedModes.includes(HYBRID_MODE) &&
		state.filter
	) {
		console.log(
			'🔄 RE-INTENTO: El modo semántico falló. Probando búsqueda híbrida...',
		);
		return 'hybridSearch';
	}

	// Si ya probamos todo o no hay alternativas
	return 'noAnswer';
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. BUILD THE GRAPH
// ─────────────────────────────────────────────────────────────────────────────
const dbPath = 'checkpoints.db';
const checkpointer = SqliteSaver.fromConnString(dbPath);
const graph = new StateGraph(AgentState)
	.addNode('routeQuestion', routeQuestion)
	.addNode('semanticSearch', semanticSearch)
	.addNode('hybridSearch', hybridSearch)
	.addNode('gradeDocuments', gradeDocuments)
	.addNode('generateAnswer', generateAnswer)
	.addNode('noAnswer', noAnswer)

	.addEdge(START, 'routeQuestion')

	.addConditionalEdges('routeQuestion', decideSearchMode, {
		semanticSearch: 'semanticSearch',
		hybridSearch: 'hybridSearch',
		noAnswer: 'noAnswer',
	})

	.addEdge('semanticSearch', 'gradeDocuments')
	.addEdge('hybridSearch', 'gradeDocuments')

	// El "Bucle": de grading puede volver a los nodos de búsqueda o avanzar
	.addConditionalEdges('gradeDocuments', decideAfterGrading, {
		generateAnswer: 'generateAnswer',
		semanticSearch: 'semanticSearch',
		hybridSearch: 'hybridSearch',
		noAnswer: 'noAnswer',
	})

	.addEdge('generateAnswer', END)
	.addEdge('noAnswer', END)
	.compile({ checkpointer });

// ─────────────────────────────────────────────────────────────────────────────
// 5. EXECUTION
// ─────────────────────────────────────────────────────────────────────────────

export async function runAgent(question: string): Promise<void> {
	// 1. Usamos un ID único para cada sesión de chat
	const thread_id = randomUUID().toString();
	const config = { configurable: { thread_id } };

	const initialState = {
		question,
		mode: '',
		filter: null,
		documents: [],
		relevantChunks: [],
		answer: '',
		triedModes: [],
	};

	console.log('\n--- 🌊 INICIANDO PASO A PASO (STREAMING) ---');

	// 2. EL PASO A PASO: Usamos .stream() para ver qué nodo se ejecuta en tiempo real
	const stream = await graph.stream(initialState, config);

	for await (const event of stream) {
		const nodeName = Object.keys(event)[0];
		const stateAtThisPoint = event[nodeName];

		console.log(`\n📍 [NODO: ${nodeName.toUpperCase()}]`);
		console.log(`   ├─ 🔍 Modo actual: ${stateAtThisPoint.mode || 'N/A'}`);
		console.log(
			`   ├─ 📚 Documentos en memoria: ${stateAtThisPoint.documents?.length || 0}`,
		);
		console.log(
			`   ├─ 🧩 Chunks relevantes: ${stateAtThisPoint.relevantChunks?.length || 0}`,
		);
		console.log(
			`   ├─ 🔄 Historial de intentos: [${stateAtThisPoint.triedModes?.join(' -> ') || ''}]`,
		);

		if (stateAtThisPoint.answer) {
			console.log(`   └─ ✅ Respuesta generada (parcial)`);
		}
		console.log(`   -------------------------------------------`);
	}

	// 3. EL HISTÓRICO: Una vez termina, consultamos la base de datos SQLite
	console.log('\n--- 📜 REBOBINANDO HISTÓRICO (Desde SQLite) ---');

	// getStateHistory devuelve un iterador con todos los estados guardados en el DB
	for await (const state of graph.getStateHistory(config)) {
		const metadata = state.metadata; // Información de quién creó el estado
		const values = state.values; // Los datos reales (AgentState)

		console.log(`\nRevisando punto de control:`);
		console.log(`- Nodo origen: ${metadata?.source || 'Inicio'}`);
		console.log(`- Pregunta: ${values.question}`);
		console.log(`- Respuesta actual: ${values.answer || '(vacía)'}`);
		console.log(
			`- Modos intentados: ${values.triedModes?.join(', ') || 'ninguno'}`,
		);
		console.log('------------------------------------------');
	}
}

async function main() {
	const pregunta =
		'¿El laboratorio de la Especialista en Flujos Etéreos estaba diseñado con paredes de vidrio plúmbeo?';
	console.log('🚀 Iniciando Agente RAG con Lógica de Reintento...');

	try {
		initDBAndChunking();
		await runAgent(pregunta);
		console.log('\n--- 🎯 RESPUESTA FINAL ---');
	} catch (error) {
		console.error('❌ Error:', error);
	}
}

main();
