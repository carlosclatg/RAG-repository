import * as lancedb from '@lancedb/lancedb';

const DB_PATH: string = './data';

let instance: lancedb.Connection; //singleton instance

export async function connectToVectorDB(): Promise<lancedb.Connection> {
	if (!instance) {
		instance = await lancedb.connect(DB_PATH);
	}
	return Promise.resolve(instance);
}

export async function openTable(
	collectionName: string,
): Promise<lancedb.Table> {
	return await instance.openTable(collectionName);
}
