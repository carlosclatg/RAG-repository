/**
 * RAG Agent built with LangGraph
 * ================================
 *
 * This file demonstrates the core LangGraph concepts by building a small
 * Retrieval-Augmented Generation agent on top of the existing RAG system.
 *
 * Graph flow:
 *
 *   [__start__]
 *       │
 *   [routeQuestion]  ── decides: "semantic" | "hybrid" | "no_retrieval"
 *       │
 *   ┌───┴──────────────────┐
 *   │                      │
 * [semanticSearch]   [hybridSearch]
 *   │                      │
 *   └──────────┬───────────┘
 *              │
 *        [gradeDocuments]  ── decides: "relevant" | "not_relevant"
 *              │
 *        ┌─────┴──────┐
 *        │            │
 *  [generateAnswer]  [noAnswer]
 *        │            │
 *     [__end__]    [__end__]
 */

import { Annotation, StateGraph, END, START } from '@langchain/langgraph';
import { generateEmbedding } from '../embedding/index.js';
import { connectToVectorDB } from '../db/index.js';
import { generateResponse, selectRAGMode, HYBRID_MODE } from '../llm/index.js';
import { rankingResponses } from '../rerank/index.js';
import { QUERY_LIMIT } from '../config/index.js';

// ─────────────────────────────────────────────────────────────────────────────
// 1. STATE DEFINITION
//    The State is the "memory" that all nodes read from and write to.
//    Every node receives the full state and returns a *partial* update.
//    LangGraph merges the update into the state automatically.
//
//    Annotation.Root() declares the shape of our state.
//    Each field can optionally have a:
//      - default()  → initial value when the field is not provided
//      - reducer()  → how incoming updates are merged (e.g. append vs replace)
// ─────────────────────────────────────────────────────────────────────────────
interface DocumentChunk {
	id: number;
	text: string;
	chapter: string;
	_distance?: number;
}

const AgentState = Annotation.Root({
	/** The original question from the user */
	question: Annotation<string>(),

	/** RAG mode chosen by the router: "semantico" | "hibrido" | "no_retrieval" */
	mode: Annotation<string>(),

	/** Filter string extracted for hybrid (FTS) search */
	filter: Annotation<string | null>(),

	/** Raw document chunks retrieved from the vector DB */
	documents: Annotation<DocumentChunk[]>({
		value: (_prev, next) => next,
		default: () => [],
	}),

	/** Chunks kept after re-ranking */
	relevantChunks: Annotation<string[]>({
		value: (_prev, next) => next,
		default: () => [],
	}),

	/** Final answer produced by the LLM */
	answer: Annotation<string>(),
});

// Convenient type alias so node functions are properly typed
type State = typeof AgentState.State;

const COLLECTION_NAME = 'documents';

// ─────────────────────────────────────────────────────────────────────────────
// 2. NODES
//    Each node is a plain async function: (state) => Partial<State>
//    Nodes must NOT mutate the state object directly — they return updates.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * NODE: routeQuestion
 * Calls the LLM router to decide which search strategy to use.
 * Updates state with `mode` and `filter`.
 */
async function routeQuestion(state: State) {
	console.log('\n[routeQuestion] Deciding search strategy...');
	try {
		const raw = await selectRAGMode(state.question);
		const { mode, filter } = JSON.parse(raw);
		console.log(`[routeQuestion] → mode: ${mode}, filter: ${filter}`);
		return { mode, filter };
	} catch {
		// Fallback to semantic search if the router fails
		console.warn('[routeQuestion] Router failed, defaulting to semantic.');
		return { mode: 'semantico', filter: null };
	}
}

/**
 * NODE: semanticSearch
 * Performs a vector-similarity search using the question embedding.
 * Updates state with `documents`.
 */
async function semanticSearch(state: State) {
	console.log('\n[semanticSearch] Running vector search...');
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);
	const embedding = await generateEmbedding(state.question);

	const results = (await table
		.search(embedding)
		.limit(QUERY_LIMIT)
		.toArray()) as DocumentChunk[];

	console.log(`[semanticSearch] Retrieved ${results.length} chunks.`);
	return { documents: results };
}

/**
 * NODE: hybridSearch
 * Performs a full-text search using the extracted proper-noun filter.
 * Falls back to semantic search if no filter was found.
 * Updates state with `documents`.
 */
async function hybridSearch(state: State) {
	console.log('\n[hybridSearch] Running FTS search...');
	const db = await connectToVectorDB();
	const table = await db.openTable(COLLECTION_NAME);

	if (!state.filter) {
		console.warn(
			'[hybridSearch] No filter found, falling back to semantic.',
		);
		return semanticSearch(state);
	}

	const results = (await table
		.search(state.filter)
		.limit(QUERY_LIMIT)
		.toArray()) as DocumentChunk[];

	console.log(`[hybridSearch] Retrieved ${results.length} chunks.`);
	return { documents: results };
}

/**
 * NODE: gradeDocuments
 * Passes retrieved chunks through the Cohere re-ranker.
 * Only chunks above the relevance threshold survive.
 * Updates state with `relevantChunks`.
 */
async function gradeDocuments(state: State) {
	console.log('\n[gradeDocuments] Re-ranking and grading documents...');
	const rawTexts = state.documents.map(
		(d) => `[Chapter: ${d.chapter}]\n${d.text}`,
	);

	const relevantChunks = await rankingResponses(state.question, rawTexts);
	console.log(
		`[gradeDocuments] ${relevantChunks.length} / ${rawTexts.length} chunks passed the threshold.`,
	);
	return { relevantChunks };
}

/**
 * NODE: generateAnswer
 * Builds the context from relevant chunks and calls the LLM to get an answer.
 * Updates state with `answer`.
 */
async function generateAnswer(state: State) {
	console.log('\n[generateAnswer] Generating answer from context...');
	const context = state.relevantChunks.join('\n---\n');
	const result = await generateResponse(state.question, context);

	if (result.success) {
		return { answer: result.data };
	}
	console.error('[generateAnswer] LLM error:', result.error);
	return { answer: `Error generating answer: ${result.error}` };
}

/**
 * NODE: noAnswer
 * Reached when no relevant documents were found.
 * Sets a fallback message in `answer`.
 */
function noAnswer(state: State) {
	console.log('\n[noAnswer] No relevant context found.');
	return {
		answer: `I couldn't find relevant information for: "${state.question}"`,
	};
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. CONDITIONAL EDGE FUNCTIONS
//    These are plain functions that receive the state and return a string
//    that LangGraph uses to select the next node.
// ─────────────────────────────────────────────────────────────────────────────

/**
 * EDGE: after routeQuestion
 * Routes to semanticSearch, hybridSearch, or noAnswer directly.
 */
function decideSearchMode(state: State): string {
	if (state.mode === HYBRID_MODE) return 'hybridSearch';
	if (state.mode === 'no_retrieval') return 'noAnswer';
	return 'semanticSearch'; // default: semantic
}

/**
 * EDGE: after gradeDocuments
 * If we have relevant chunks → generate; otherwise → noAnswer.
 */
function decideAfterGrading(state: State): string {
	return state.relevantChunks.length > 0 ? 'generateAnswer' : 'noAnswer';
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. BUILD THE GRAPH
//    Steps:
//      a) Create a StateGraph with the state shape
//      b) Add nodes (name → function)
//      c) Wire edges: fixed (addEdge) and conditional (addConditionalEdges)
//      d) Set the entry point (START → first node)
//      e) Compile — this validates the graph and returns a runnable object
// ─────────────────────────────────────────────────────────────────────────────

const graph = new StateGraph(AgentState)
	// ── Register nodes ────────────────────────────────────────────────────────
	.addNode('routeQuestion', routeQuestion)
	.addNode('semanticSearch', semanticSearch)
	.addNode('hybridSearch', hybridSearch)
	.addNode('gradeDocuments', gradeDocuments)
	.addNode('generateAnswer', generateAnswer)
	.addNode('noAnswer', noAnswer)

	// ── Entry point ───────────────────────────────────────────────────────────
	.addEdge(START, 'routeQuestion')

	// ── After routing: conditional fan-out ───────────────────────────────────
	.addConditionalEdges('routeQuestion', decideSearchMode, {
		semanticSearch: 'semanticSearch',
		hybridSearch: 'hybridSearch',
		noAnswer: 'noAnswer',
	})

	// ── Both search branches converge into grading ────────────────────────────
	.addEdge('semanticSearch', 'gradeDocuments')
	.addEdge('hybridSearch', 'gradeDocuments')

	// ── After grading: conditional fork to generate or give up ────────────────
	.addConditionalEdges('gradeDocuments', decideAfterGrading, {
		generateAnswer: 'generateAnswer',
		noAnswer: 'noAnswer',
	})

	// ── Terminal edges ────────────────────────────────────────────────────────
	.addEdge('generateAnswer', END)
	.addEdge('noAnswer', END)

	// ── Compile ───────────────────────────────────────────────────────────────
	.compile();

// ─────────────────────────────────────────────────────────────────────────────
// 5. ENTRY POINT
//    graph.invoke() runs the compiled graph from START to END.
//    It receives the initial state and returns the final state.
// ─────────────────────────────────────────────────────────────────────────────
export async function runAgent(question: string): Promise<string> {
	const finalState = await graph.invoke({
		question,
		mode: '',
		filter: null,
		documents: [],
		relevantChunks: [],
		answer: '',
	});
	return finalState.answer;
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. VISUALISE (optional helper)
//    graph.getGraph().toJSON() exposes the graph structure.
//    Uncomment the Mermaid export below to get a diagram of the graph.
// ─────────────────────────────────────────────────────────────────────────────
export async function printGraphDiagram(): Promise<void> {
	const diagram = graph.getGraph().toJSON();
	console.log('\n── Graph nodes ──');
	diagram.nodes.forEach((n: { id: string }) => console.log(' •', n.id));
	console.log('\n── Graph edges ──');
	diagram.edges.forEach(
		(e: { source: string; target: string; conditional: boolean }) =>
			console.log(
				` ${e.source} ──${e.conditional ? '[cond]' : '──────'}→ ${e.target}`,
			),
	);
}
