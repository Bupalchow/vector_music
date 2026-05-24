import {createIndex} from './config/db.js';

const indexName = process.env.PINECONE_INDEX_NAME;

const initialize = async () => {
    await createIndex(indexName);
}

initialize();