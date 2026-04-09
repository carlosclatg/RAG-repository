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
import { logInfo, logWarn, logError, shutdownLogger } from '../observability/index.ts';
import express from 'express';
import type { Request, Response } from 'express';

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
	// Observability: Distributed Tracing
	traceId: Annotation<string>(),
	spanId: Annotation<string>(),
});

type State = typeof AgentState.State;
const COLLECTION_NAME = 'documents';

// ─────────────────────────────────────────────────────────────────────────────
// 1.5 OBSERVABILITY HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Generates a unique trace ID for distributed tracing
 */
function generateTraceId(): string {
	return randomUUID().replace(/-/g, '').substring(0, 32);
}

/**
 * Generates a unique span ID for distributed tracing
 */
function generateSpanId(): string {
	return randomUUID().replace(/-/g, '').substring(0, 16);
}

/**
 * Enhanced logging with trace and span IDs
 */
function logWithTrace(
	message: string,
	traceId: string,
	spanId: string,
	level: 'info' | 'warn' | 'error' = 'info',
	metadata: Record<string, any> = {},
) {
	const enrichedMetadata = {
		trace_id: traceId,
		span_id: spanId,
		...metadata,
	};

	if (level === 'info') {
		logInfo(message, enrichedMetadata);
	} else if (level === 'warn') {
		logWarn(message, enrichedMetadata);
	} else if (level === 'error') {
		logError(`${message}`, enrichedMetadata);
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. NODES
// ─────────────────────────────────────────────────────────────────────────────

async function routeQuestion(state: State) {
	const nodeSpanId = generateSpanId();
	logWithTrace('[routeQuestion] Decidiendo estrategia inicial...', state.traceId, nodeSpanId, 'info');
	try {
		const raw = await selectRAGMode(state.question);
		const { mode, filter } = JSON.parse(raw);
		logWithTrace(`[routeQuestion] → modo sugerido: ${mode}`, state.traceId, nodeSpanId, 'info', {
			mode,
			filter: filter ? 'present' : 'null',
		});
		return { mode, filter };
	} catch (error) {
		const errorMsg = error instanceof Error ? error.message : String(error);
		logWithTrace(`[routeQuestion] Error parsing mode, using default`, state.traceId, nodeSpanId, 'warn', {
			error: errorMsg,
		});
		return { mode: 'semantico', filter: null };
	}
}

async function semanticSearch(state: State) {
	const nodeSpanId = generateSpanId();
	logWithTrace('[semanticSearch] Ejecutando búsqueda vectorial...', state.traceId, nodeSpanId, 'info');
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);
	const embedding = await generateEmbedding(state.question);

	const results = (await table
		.search(embedding)
		.limit(QUERY_LIMIT)
		.toArray()) as DocumentChunk[];

	logWithTrace(`[semanticSearch] Encontrados ${results.length} chunks.`, state.traceId, nodeSpanId, 'info', {
		count: results.length,
		queryLimit: QUERY_LIMIT,
	});
	return {
		documents: results,
		triedModes: ['semantico'],
	};
}

async function hybridSearch(state: State) {
	const nodeSpanId = generateSpanId();
	logWithTrace('[hybridSearch] Ejecutando búsqueda FTS (Híbrida)...', state.traceId, nodeSpanId, 'info', {
		hasFilter: !!state.filter,
	});
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);

	if (!state.filter) {
		logWithTrace('[hybridSearch] Sin filtro FTS, redirigiendo a semántico.', state.traceId, nodeSpanId, 'warn');
		return semanticSearch(state);
	}

	const results = (await table
		.search(state.filter)
		.limit(QUERY_LIMIT)
		.toArray()) as DocumentChunk[];

	logWithTrace(`[hybridSearch] Encontrados ${results.length} chunks.`, state.traceId, nodeSpanId, 'info', {
		count: results.length,
		queryLimit: QUERY_LIMIT,
	});
	return {
		documents: results,
		triedModes: [HYBRID_MODE],
	};
}

async function gradeDocuments(state: State) {
	const nodeSpanId = generateSpanId();
	logWithTrace('[gradeDocuments] Re-rankeando y evaluando relevancia...', state.traceId, nodeSpanId, 'info', {
		documentsCount: state.documents.length,
	});
	if (state.documents.length === 0) {
		logWithTrace('[gradeDocuments] Sin documentos para evaluar', state.traceId, nodeSpanId, 'warn');
		return { relevantChunks: [] };
	}

	const rawTexts = state.documents.map(
		(d) => `[Chapter: ${d.chapter}]\n${d.text}`,
	);

	const relevantChunks = await rankingResponses(state.question, rawTexts);
	logWithTrace(
		`[gradeDocuments] ${relevantChunks.length} chunks pasaron el filtro.`,
		state.traceId,
		nodeSpanId,
		'info',
		{
			relevantCount: relevantChunks.length,
			totalCount: state.documents.length,
			filterRatio: (relevantChunks.length / state.documents.length).toFixed(2),
		},
	);
	return { relevantChunks };
}

async function generateAnswer(state: State) {
	const nodeSpanId = generateSpanId();
	logWithTrace('[generateAnswer] Generando respuesta final...', state.traceId, nodeSpanId, 'info', {
		relevantChunksCount: state.relevantChunks.length,
	});
	const context = state.relevantChunks.join('\n---\n');
	const result = await generateResponse(state.question, context);

	const answerLength = result.success ? result.data.length : 0;
	logWithTrace(
		`[generateAnswer] Respuesta generada (${answerLength} caracteres)`,
		state.traceId,
		nodeSpanId,
		result.success ? 'info' : 'error',
		{
			answerLength,
			success: result.success,
			errorMsg: result.error || undefined,
		},
	);

	return { answer: result.success ? result.data : `Error: ${result.error}` };
}

function noAnswer(state: State) {
	const nodeSpanId = generateSpanId();
	logWithTrace('[noAnswer] Agotadas todas las vías de búsqueda.', state.traceId, nodeSpanId, 'warn', {
		question: state.question,
		triedModes: state.triedModes.join(', '),
	});
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
	const decisionSpanId = generateSpanId();

	if (state.relevantChunks.length > 0) {
		logWithTrace('[decideAfterGrading] Documentos relevantes encontrados → generateAnswer', state.traceId, decisionSpanId, 'info');
		return 'generateAnswer';
	}

	// Si falló el híbrido pero no hemos probado el semántico
	if (
		state.triedModes.includes(HYBRID_MODE) &&
		!state.triedModes.includes('semantico')
	) {
		logWithTrace(
			'[decideAfterGrading] RE-INTENTO: Probando búsqueda semántica',
			state.traceId,
			decisionSpanId,
			'info',
			{ previousMode: HYBRID_MODE },
		);
		return 'semanticSearch';
	}

	// Si falló el semántico pero el router nos dio un filtro y no hemos probado híbrido
	if (
		state.triedModes.includes('semantico') &&
		!state.triedModes.includes(HYBRID_MODE) &&
		state.filter
	) {
		logWithTrace(
			'[decideAfterGrading] RE-INTENTO: Probando búsqueda híbrida',
			state.traceId,
			decisionSpanId,
			'info',
			{ previousMode: 'semantico' },
		);
		return 'hybridSearch';
	}

	// Si ya probamos todo o no hay alternativas
	logWithTrace(
		'[decideAfterGrading] Todas las estrategias agotadas → noAnswer',
		state.traceId,
		decisionSpanId,
		'warn',
		{ triedModes: state.triedModes.join(', ') },
	);
	return 'noAnswer';
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. BUILD THE GRAPH
// ─────────────────────────────────────────────────────────────────────────────
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dbPath = path.resolve(__dirname, './checkpoints.db');
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

export async function runAgent(question: string, traceId?: string): Promise<string> {
	// 1. Usamos un ID único para cada sesión de chat
	const trace_id = traceId || generateTraceId();
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
		traceId: trace_id,
		spanId: generateSpanId(),
	};

	logWithTrace(
		'[INICIO] Iniciando ejecución del agente RAG',
		trace_id,
		initialState.spanId,
		'info',
		{
			threadId: thread_id,
			question: question.substring(0, 100),
		},
	);

	// 2. EL PASO A PASO: Usamos .stream() para ver qué nodo se ejecuta en tiempo real
	const stream = await graph.stream(initialState, config);

	let finalAnswer = '';

	for await (const event of stream) {
		const nodeName = Object.keys(event)[0];
		const stateAtThisPoint = event[nodeName];

		logWithTrace(
			`[NODO: ${nodeName.toUpperCase()}]`,
			trace_id,
			stateAtThisPoint.spanId,
			'info',
			{
				node: nodeName,
				mode: stateAtThisPoint.mode || 'N/A',
				documents: stateAtThisPoint.documents?.length || 0,
				relevantChunks: stateAtThisPoint.relevantChunks?.length || 0,
				triedModes: stateAtThisPoint.triedModes?.join(' -> ') || '',
				hasAnswer: !!stateAtThisPoint.answer,
			},
		);

		// Capture the final answer when it's generated
		if (stateAtThisPoint.answer) {
			finalAnswer = stateAtThisPoint.answer;
		}
	}

	// 3. EL HISTÓRICO: Una vez termina, consultamos la base de datos SQLite
	logWithTrace('REBOBINANDO HISTÓRICO (Desde SQLite)', trace_id, generateSpanId(), 'info');

	// getStateHistory devuelve un iterador con todos los estados guardados en el DB
	for await (const state of graph.getStateHistory(config)) {
		const metadata = state.metadata; // Información de quién creó el estado
		const values = state.values; // Los datos reales (AgentState)

		logWithTrace(
			'Checkpoint de estado',
			trace_id,
			values.spanId,
			'info',
			{
				source: metadata?.source || 'Inicio',
				question: values.question,
				answer: values.answer ? `${values.answer.substring(0, 50)}...` : '(vacía)',
				triedModes: values.triedModes?.join(', ') || 'ninguno',
			},
		);
	}

	logWithTrace(
		'[FIN] Agente RAG completó ejecución',
		trace_id,
		generateSpanId(),
		'info',
		{
			answerLength: finalAnswer.length,
		},
	);

	return finalAnswer;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. EXPRESS API SERVER
// ─────────────────────────────────────────────────────────────────────────────

const app = express();
const PORT = 8080;

// Middleware
app.use(express.json());

// Health check endpoint
app.get('/health', (_req: Request, res: Response) => {
	const healthTraceId = generateTraceId();
	const healthSpanId = generateSpanId();

	logWithTrace('Health check request', healthTraceId, healthSpanId, 'info', {
		endpoint: '/health',
	});

	res.json({
		status: 'OK',
		timestamp: new Date().toISOString(),
		trace_id: healthTraceId,
		span_id: healthSpanId,
	});
});

// RAG Agent endpoint
app.post('/api/ask', async (req: Request, res: Response) => {
	const apiSpanId = generateSpanId();
	const apiTraceId = generateTraceId();

	try {
		const { question, message } = req.body;
		const userQuestion = question || message;

		if (!userQuestion || typeof userQuestion !== 'string') {
			logWithTrace(
				'API Request validation failed',
				apiTraceId,
				apiSpanId,
				'warn',
				{
					reason: 'Missing or invalid question/message',
				},
			);
			return res.status(400).json({
				error: 'Missing or invalid question/message field',
				example: { question: 'Your question here?' },
				trace_id: apiTraceId,
				span_id: apiSpanId,
			});
		}

		logWithTrace(
			'API Request received',
			apiTraceId,
			apiSpanId,
			'info',
			{
				question: userQuestion.substring(0, 100),
				userAgent: req.get('user-agent'),
				ip: req.ip,
			},
		);

		const answer = await runAgent(userQuestion, apiTraceId);

		logWithTrace(
			'API Response generated successfully',
			apiTraceId,
			apiSpanId,
			'info',
			{
				answerLength: answer.length,
				httpStatus: 200,
			},
		);

		res.json({
			success: true,
			question: userQuestion,
			answer,
			timestamp: new Date().toISOString(),
			trace_id: apiTraceId,
			span_id: apiSpanId,
		});
	} catch (error) {
		const errorMessage = error instanceof Error ? error.message : String(error);
		const errorStack = error instanceof Error ? error.stack : undefined;

		logWithTrace(
			'API Request failed',
			apiTraceId,
			apiSpanId,
			'error',
			{
				error: errorMessage,
				stack: errorStack ? errorStack.substring(0, 500) : undefined,
				httpStatus: 500,
			},
		);

		res.status(500).json({
			success: false,
			error: errorMessage,
			timestamp: new Date().toISOString(),
			trace_id: apiTraceId,
			span_id: apiSpanId,
		});
	}
});

// Start server
async function startServer() {
	const startupTraceId = generateTraceId();
	const startupSpanId = generateSpanId();

	try {
		logWithTrace('Initializing RAG system...', startupTraceId, startupSpanId, 'info');
		await initDBAndChunking();
		logWithTrace('RAG system initialized successfully', startupTraceId, startupSpanId, 'info');

		app.listen(PORT, () => {
			logWithTrace(`Express server running on http://localhost:${PORT}`, startupTraceId, startupSpanId, 'info', {
				port: PORT,
			});
			logWithTrace(`POST /api/ask - Send {"question": "..."} to query the RAG agent`, startupTraceId, startupSpanId, 'info');
			logWithTrace(`GET /health - Check server health`, startupTraceId, startupSpanId, 'info');
			console.log('\n=== RAG Server Started ===');
			console.log(`Trace ID for startup: ${startupTraceId}`);
			console.log(`Ready to receive requests at http://localhost:${PORT}\n`);
		});
	} catch (error) {
		const errorMsg = error instanceof Error ? error.message : String(error);
		logWithTrace(
			`Failed to start server`,
			startupTraceId,
			startupSpanId,
			'error',
			{
				error: errorMsg,
			},
		);
		await shutdownLogger();
		process.exit(1);
	}
}

// Handle graceful shutdown
process.on('SIGINT', async () => {
	const shutdownTraceId = generateTraceId();
	const shutdownSpanId = generateSpanId();
	logWithTrace('Shutting down gracefully (SIGINT)...', shutdownTraceId, shutdownSpanId, 'info');
	await shutdownLogger();
	process.exit(0);
});

process.on('SIGTERM', async () => {
	const shutdownTraceId = generateTraceId();
	const shutdownSpanId = generateSpanId();
	logWithTrace('Shutting down gracefully (SIGTERM)...', shutdownTraceId, shutdownSpanId, 'info');
	await shutdownLogger();
	process.exit(0);
});

startServer();
