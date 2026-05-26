import { randomUUID } from 'node:crypto';
import { Pinecone } from '@pinecone-database/pinecone';

const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
});

const indexName = process.env.PINECONE_INDEX_NAME;
const namespace = process.env.PINECONE_NAMESPACE ?? 'default';
const geminiApiKey = process.env.GEMINI_API_KEY;
const geminiModel = process.env.GEMINI_EMBEDDING_MODEL ?? 'gemini-embedding-2';
const embeddingDimension = Number(process.env.GEMINI_EMBEDDING_DIMENSION ?? 512);

const getEmbeddingText = (processedAudioText) => {
    return `task: classification | query: ${processedAudioText}`;
};

const extractEmbeddingValues = (responseBody) => {
    if (Array.isArray(responseBody?.embeddings) && responseBody.embeddings.length) {
        return responseBody.embeddings[0]?.values ?? responseBody.embeddings[0]?.embedding?.values ?? null;
    }

    return responseBody?.embedding?.values ?? null;
};

export const createIndex = async (name = indexName) => {
    try {
        await pc.createIndex({
            name,
            dimension: embeddingDimension,
            spec: {
                serverless: {
                    cloud: 'aws',
                    region: 'us-east-1',
                },
            },
        });
        console.log(`Index ${name} created successfully.`);
    } catch (error) {
        console.error('Error creating index:', error);
    }
};

export const createGeminiEmbedding = async (processedAudioText) => {
    if (!geminiApiKey) {
        throw new Error('GEMINI_API_KEY is required to generate embeddings.');
    }

    const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${geminiModel}:embedContent?key=${geminiApiKey}`,
        {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                content: {
                    parts: [
                        {
                            text: getEmbeddingText(processedAudioText),
                        },
                    ],
                },
                outputDimensionality: embeddingDimension,
            }),
        }
    );

    const responseBody = await response.json();

    if (!response.ok) {
        throw new Error(
            `Gemini embedding request failed: ${response.status} ${response.statusText} - ${JSON.stringify(responseBody)}`
        );
    }

    const embeddingValues = extractEmbeddingValues(responseBody);

    if (!Array.isArray(embeddingValues) || !embeddingValues.length) {
        throw new Error('Gemini did not return a valid embedding vector.');
    }

    return embeddingValues;
};

export const upsertProcessedAudioEmbedding = async ({
    id = randomUUID(),
    processedAudioText,
    metadata = {},
} = {}) => {
    if (!indexName) {
        throw new Error('PINECONE_INDEX_NAME is required to upsert embeddings.');
    }

    const values = await createGeminiEmbedding(processedAudioText);
    const index = pc.index(indexName).namespace(namespace);

    await index.upsert([
        {
            id,
            values,
            metadata: {
                processedAudioText,
                ...metadata,
            },
        },
    ]);

    return {
        id,
        namespace,
        indexName,
        vectorSize: values.length,
    };
};