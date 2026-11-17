import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null;

const API_BASE_URL = self.location.port === '3000' 
    ? 'http://localhost:3002/api' 
    : `${self.location.origin}/api`;

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
};

/*===================================================================================================
                                   SETUP DO DATASET PARA TREINAMENTO
===================================================================================================*/

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    // Normalizando os dados
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const colors = [...new Set(products.map(p => p.color))];
    const categories = [...new Set(products.map(p => p.category))];

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => [color, index])
    );

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    );

    // Computar a média de idade dos compradores por produto
    const midAge = (minAge + maxAge) / 2;
    const ageSums = {};
    const ageCounts = {};

    users.forEach(user => {
        (user.purchases || []).forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        });
    });

    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge;

            return [product.name, normalize(avg, minAge, maxAge)];
        })
    );

    return {
        products,
        users,
        colorsIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        productAvgAgeNorm,
        numCategories: categories.length,
        numColors: colors.length,
        dimentions: 2 + categories.length + colors.length
    };
}

const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

function encodeProduct(product, context) {
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ]);

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ]);

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    );

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    );

    return tf.concat1d([price, age, category, color]);
}

function encodeUser(user, context) {
    if (user.purchases && user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ]);
    }

    return tf.concat1d([
        tf.zeros([1]), // O preço é ignorado
        tf.tensor1d([
            normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
        ]),
        tf.zeros([context.numCategories]), // Categorias ignoradas
        tf.zeros([context.numColors]), // Cores ignoradas
    ]).reshape([1, context.dimentions]);
}

function createTrainingData(context) {
    const inputs = [];
    const labels = [];
    context.users
        .filter(u => u.purchases && u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync();
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync();

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ? 1 : 0
                );

                inputs.push([...userVector, ...productVector]);
                labels.push(label);
            });
        });

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
    };
}

/*===================================================================================================
        Configuração e Treinamento da Rede Neural
===================================================================================================*/
async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential();

    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    );

    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    );

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(trainData.xs, trainData.ys, {
        epochs: 100,
        batchSize: 32,
        shuffle: true,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                postMessage({
                    type: workerEvents.trainingLog,
                    epoch: epoch,
                    loss: logs.loss,
                    accuracy: logs.acc
                });
            }
        }
    });

    return model;
}

async function trainModel({ users }) {
    console.log('Training model with users:', users);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 30 } });

    // 1. Obter produtos da API do banco de dados vetorial
    let products = [];
    try {
        const prodRes = await fetch(`${API_BASE_URL}/products`);
        if (prodRes.ok) {
            products = await prodRes.json();
        } else {
            throw new Error('Falha ao obter produtos da API');
        }
    } catch (err) {
        console.warn('[Worker] Usando fallback local para /data/products.json:', err.message);
        products = await (await fetch('/data/products.json')).json();
    }

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    // 2. Gerar o contexto e os vetores de cada produto
    const context = makeContext(products, users);
    context.productVectors = products.map(product => {
        return {
            id: product.id,
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        };
    });

    _globalCtx = context;

    // 3. Persistir os embeddings gerados no Banco de Dados Vetorial!
    try {
        const itemsToSave = context.productVectors.map(pv => ({
            id: pv.id,
            vector: Array.from(pv.vector)
        }));
        await fetch(`${API_BASE_URL}/products/embeddings`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ items: itemsToSave })
        });
        console.log('[Worker] Embeddings gravados com sucesso no Banco de Dados Vetorial!');
    } catch (err) {
        console.warn('[Worker] Aviso ao salvar embeddings no banco:', err.message);
    }

    // 4. Treinar a rede neural
    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

async function recommend(user, ctx) {
    if (!_model) return;

    const context = _globalCtx;

    // Converte o usuário fornecido no vetor de features codificadas
    const userVector = encodeUser(user, context).dataSync();

    // =========================================================================
    // ETAPA 1: BUSCA NO BANCO DE DADOS VETORIAL (Vector Database Retrieval)
    // Busca no banco os produtos mais similares ao vetor do usuário via SQL
    // =========================================================================
    let candidates = [];
    try {
        const response = await fetch(`${API_BASE_URL}/recommendations/vector-search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                userVector: Array.from(userVector),
                limit: context.productVectors.length
            })
        });

        if (response.ok) {
            const result = await response.json();
            candidates = result.candidates || [];
            console.log(`[Worker] Banco vetorial retornou ${candidates.length} produtos candidatos.`);
        }
    } catch (err) {
        console.warn('[Worker] Busca vetorial na API não alcançada, usando produtos locais:', err);
    }

    // Mapeia os produtos a serem avaliados pela rede neural
    let productsToScore = [];
    if (candidates.length > 0) {
        productsToScore = candidates.map(c => {
            const found = context.productVectors.find(pv => pv.id === c.id || pv.name === c.name);
            return {
                id: c.id,
                name: c.name,
                meta: { ...c },
                vector: found ? found.vector : (c.embedding ? new Float32Array(c.embedding) : null),
                vectorSimilarity: c.vectorSimilarity
            };
        }).filter(p => p.vector);
    } else {
        productsToScore = context.productVectors;
    }

    // =========================================================================
    // ETAPA 2: RE-RANKING COM REDE NEURAL (TensorFlow.js Re-ranking)
    // A rede neural refinada calcula a pontuação final para cada candidato
    // =========================================================================
    const inputs = productsToScore.map(({ vector }) => {
        return [...userVector, ...vector];
    });

    const inputTensor = tf.tensor2d(inputs);
    const predictions = _model.predict(inputTensor);
    const scores = predictions.dataSync();

    const recommendations = productsToScore.map((item, index) => {
        return {
            ...item.meta,
            id: item.id,
            name: item.name,
            score: scores[index], // Predição da rede neural (0.00 a 1.00)
            vectorSimilarity: item.vectorSimilarity !== undefined ? item.vectorSimilarity : null // Similaridade vetorial do banco
        };
    });

    const sortedItems = recommendations.sort((a, b) => b.score - a.score);

    console.log('[Worker] Recomendações prontas para:', user.name);

    postMessage({
        type: workerEvents.recommend,
        user,
        recommendations: sortedItems
    });
}

const handlers = {
    [workerEvents.trainModel]: trainModel,
    [workerEvents.recommend]: d => recommend(d.user, _globalCtx),
};

self.onmessage = async e => {
    const { action, ...data } = e.data;
    if (handlers[action]) await handlers[action](data);
};
