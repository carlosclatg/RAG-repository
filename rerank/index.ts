// Rerank API (v2) (POST /v2/rerank)
const THRESHOLD = 0.8;
export async function rankingResponses(query: string, possibleResponses: string[], top_n: number = 2): Promise<string[]> {
    const bodyBuild = JSON.stringify({
        "model": "rerank-v4.0-pro",
        "query": query,
        "documents": possibleResponses,
        "top_n": 3
    })
    const apiKey = process.env.COHERE_API_KEY;
    if (!apiKey) {
        throw new Error("Missing env variable COHERE_API_KEY");
    }
    const response = await fetch("https://api.cohere.com/v2/rerank", {
    method: "POST",
    headers: {
        "Authorization": apiKey,
        "Content-Type": "application/json"
    },
    body: bodyBuild,
    });
    const body = await response.json();

    if (response.status != 200) {
        return Promise.reject(body);
    }
    
    const bestIndexs = body.results
        .filter((result: any) => result.relevance_score > THRESHOLD)
        .map((result: any) => result.index);
    const bestResponses = possibleResponses.filter((response, index) => bestIndexs.includes(index));

    return Promise.resolve(bestResponses);
}
