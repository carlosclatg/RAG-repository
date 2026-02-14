import * as lancedb from "vectordb";

const DB_PATH: string = "./data";

let instance: lancedb.Connection; //singleton instance

export async function connectToVectorDB(): Promise<lancedb.Connection> {
    if (!instance) {
        instance = await lancedb.connect(DB_PATH);
    }
    return Promise.resolve(instance);
}

export async function openTable(collectionName: string) : Promise<lancedb.Table> {
    return await instance.openTable(collectionName);

}

export async function tableSearch(table: lancedb.Table<number[]>, 
    queryEmbedding: number[], limit = 3): Promise<Record<string, unknown>[]> {
   
    return table.search(queryEmbedding).limit(limit).execute();
}