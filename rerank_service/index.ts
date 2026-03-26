// Rerank API (v2) (POST /v2/rerank)
import { COHERE_API_KEY, RERANK_THRESHOLD } from '../config/index.js';

export async function rankingResponses(
	query: string,
	possibleResponses: string[],
	top_n: number = 2,
): Promise<string[]> {
	if (!COHERE_API_KEY) {
		return possibleResponses;
	}

	const body = JSON.stringify({
		model: 'rerank-v4.0-pro',
		query,
		documents: possibleResponses,
		top_n,
	});

	const response = await fetch('https://api.cohere.com/v2/rerank', {
		method: 'POST',
		headers: {
			Authorization: `Bearer ${COHERE_API_KEY}`,
			'Content-Type': 'application/json',
		},
		body,
	});

	const responseBody = await response.json();

	if (response.status !== 200) {
		return Promise.reject(
			new Error(
				`Cohere rerank failed [${response.status}]: ${JSON.stringify(responseBody)}`,
			),
		);
	}

	const bestIndexes: number[] = responseBody.results
		.filter(
			(result: { relevance_score: number }) =>
				result.relevance_score > RERANK_THRESHOLD,
		)
		.map((result: { index: number }) => result.index);

	return possibleResponses.filter((_, index) => bestIndexes.includes(index));
}
