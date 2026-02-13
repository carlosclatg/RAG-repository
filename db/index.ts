import * as lancedb from "vectordb";

const DB_PATH: string = "./data";

let instance: lancedb.Connection;

/**
 * Establece una conexión con la base de datos local de LanceDB.
 * @returns Una promesa que resuelve en una instancia de Connection.
 */
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