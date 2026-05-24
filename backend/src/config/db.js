import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

export const createIndex = async (indexName) => {
    try {
        await pc.createIndex({
            name: indexName,
            dimension: 512,
            spec: {
                serverless: {
                    cloud: 'aws',
                    region: 'us-east-1',
                },
            },
        });
        console.log(`Index ${indexName} created successfully.`);
    } catch (error) {
        console.error('Error creating index:', error);
    }
}