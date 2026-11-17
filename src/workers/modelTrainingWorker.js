import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized for Bakery Dataset');
let _globalCtx = {};
let _model = null;

// Base da API para coletar os dados do Banco de Dados Vetorial 
const API_BASE_URL = self.location.port === '3000'
    ? 'http://localhost:3002/api'
    : `${self.location.origin}/api`;

// Definição de pesos
const WEIGHTS = {
    category: 0.4,
    storage: 0.3,
    price: 0.2,
    age: 0.1
};

/*===================================================================================================
                       SETUP DO DATASET DA PADARIA PARA TREINAMENTO
===================================================================================================*/

const normalize = (value, min, max) => (value - min) / ((max - min) || 1);

/**
 * Realiza o Pré-Processamento e Engenharia de Recursos (Feature Engineering).
 * 
 * Analisa todo o universo de produtos e usuários para construir o "Contexto Global"
 * necessário para normalizar números e criar os índices de One-Hot Encoding.
 * 
 * Principais responsabilidades:
 * 1. Mapear os extremos (min/max) de preço e idade para o Min-Max Scaling (escala 0 a 1).
 * 2. Criar dicionários de índices (vocabulário) para categorias e tipos de armazenamento.
 * 3. Calcular a média de idade dos compradores por produto (agregação comportamental).
 * 4. Determinar dinamicamente a dimensionalidade dos vetores para a Rede Neural.
 * 
 * @param {Array<Object>} products - Lista de produtos obtida do Banco de Dados
 * @param {Array<Object>} users - Lista de usuários e seus históricos de compras do BD
 * @returns {Object} Objeto de contexto com regras de normalização, índices e estatísticas
 */
function makeContext(products, users) {
    const ages = users.map(u => u.age);
    const prices = products.map(p => p.price);

    // Normalizando os dados
    const minAge = Math.min(...ages);
    const maxAge = Math.max(...ages);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    const storages = [...new Set(products.map(p => p.storage || p.color || 'Standard'))];
    const categories = [...new Set(products.map(p => p.category))];

    const storagesIndex = Object.fromEntries(
        storages.map((storage, index) => [storage, index])
    );

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => [category, index])
    );

    // Computar a média de idade dos compradores por produto de padaria
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
        storagesIndex,
        categoriesIndex,
        minAge,
        maxAge,
        minPrice,
        maxPrice,
        productAvgAgeNorm,
        numCategories: categories.length,
        numStorages: storages.length,
        dimentions: 2 + categories.length + storages.length
    };
}

// 
const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight);

/**
 * Converte um produto (dados brutos como texto, preço e categoria) em um 
 * VETOR NUMÉRICO (Embedding) contínuo, normalizado entre 0 e 1 e ponderado, pois
 * cada atributo possui um peso diferente, ex: preço poderá ter um peso maior do que a idade, na 
 * preferência da pessoa.
 * 
 * Esse vetor numérico tem dois papéis centrais no sistema:
 * 1. É o "Embedding" salvo na coluna vetorial do Banco de Dados para busca por similaridade (KNN/Cosseno).
 * 2. É combinado com o vetor do usuário [userVector + productVector] para alimentar a Rede Neural.
 * 
 * Estrutura do vetor resultante:
 * [ preço_norm*peso, idade_média_norm*peso, categoria_one_hot*peso, armazenamento_one_hot*peso ]
 *
 * @param {Object} product - Dados do produto ({ name, price, category, storage })
 * @param {Object} context - Metadados de normalização (min/max preços, índices one-hot)
 * @returns {tf.Tensor1D} Tensor unidimensional com o embedding ponderado do produto
 */
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
        context.categoriesIndex[product.category] ?? 0,
        context.numCategories,
        WEIGHTS.category
    );

    const storageVal = product.storage || product.color || 'Standard';
    const storage = oneHotWeighted(
        context.storagesIndex[storageVal] ?? 0,
        context.numStorages,
        WEIGHTS.storage
    );

    return tf.concat1d([price, age, category, storage]);
}

/**
 * Converte um USUÁRIO em um VETOR NUMÉRICO (Embedding do Usuário) no mesmo 
 * espaço dimensional dos produtos (mesma quantidade de dimensões e escalas).
 * 
 * Essa função trata dois cenários fundamentais em recomendação:
 * 
 * 1. USUÁRIO COM HISTÓRICO:
 *    Calcula o "centro de massa" (centróide / média vetorial) de todos os produtos 
 *    que o usuário já comprou. O vetor resultante representa o "gosto médio" do cliente
 *    (suas categorias preferidas, faixa de preço média e tipo de armazenamento usual).
 * 
 * 2. USUÁRIO NOVO (Cold-Start Problem - Ex: "Josézin da Silva"):
 *    Quando o usuário não tem compras anteriores, o modelo não sabe suas preferências
 *    de preço ou categoria. Nesse caso, gera um vetor baseado EXCLUSIVAMENTE na sua idade,
 *    preenchendo os outros atributos com zeros. A Rede Neural usará a idade para aproximá-lo
 *    de produtos tipicamente consumidos por pessoas da mesma faixa etária.
 * 
 * @param {Object} user - Dados do usuário ({ id, name, age, purchases: [...] })
 * @param {Object} context - Metadados e limites globais gerados por makeContext
 * @returns {tf.Tensor2D} Tensor no formato [1, dimentions] representando o perfil do usuário
 */
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
        tf.zeros([1]), // Preço ignorado no perfil base
        tf.tensor1d([
            normalize(user.age, context.minAge, context.maxAge) * WEIGHTS.age
        ]),
        tf.zeros([context.numCategories]), // Categorias ignoradas
        tf.zeros([context.numStorages]),   // Armazenamento ignorado
    ]).reshape([1, context.dimentions]);
}

/**
 * Constrói o Dataset de Treinamento supervisionado para a Rede Neural (Tensores xs e ys).
 * 
 * Transforma a recomendação em uma tarefa de CLASSIFICAÇÃO BINÁRIA:
 * "Dado o par (Usuário X, Produto Y), qual é a probabilidade dele comprar o produto?"
 * 
 * Como o dataset é gerado:
 * 1. Cruza todos os usuários que têm histórico com todos os produtos do catálogo (Produto Cartesiano).
 * 2. Para cada par, concatena [Vetor do Usuário (N dimensões) + Vetor do Produto (N dimensões)].
 *    O vetor resultante de entrada terá tamanho 2 * N (inputDimention).
 * 3. Define o rótulo (Label / Ground Truth):
 *    - 1 (Positivo): O usuário comprou esse produto no histórico.
 *    - 0 (Negativo): O usuário não comprou esse produto.
 * 
 * @param {Object} context - Objeto de contexto contendo usuários, produtos e dimensões
 * @returns {Object} { xs: Tensor2D das entradas, ys: Tensor2D dos rótulos (0 ou 1), inputDimention }
 */
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

    const epochs = 40;
    await model.fit(trainData.xs, trainData.ys, {
        epochs,
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

/**
 * Orquestrador do Pipeline de Treinamento e Indexação Vetorial.
 * 
 * Executa de forma assíncrona dentro de um Web Worker (thread secundária), garantindo
 * que cálculos pesados de Tensores e épocas de treino não congelem a interface (UI).
 * 
 * Etapas da esteira (Pipeline):
 * 1. OBTENÇÃO DOS DADOS: Consome o catálogo de produtos diretamente do Banco de Dados via API REST.
 * 2. FEATURE ENGINEERING: Gera o contexto estatístico (makeContext) e os Embeddings dos produtos (encodeProduct).
 * 3. INDEXAÇÃO VETORIAL: Salva os embeddings calculados na coluna vetorial do Banco de Dados (POST /api/products/embeddings).
 * 4. TREINAMENTO: Constrói a matriz de pares (createTrainingData) e ajusta os pesos da Rede Neural via Backpropagation.
 * 5. FEEDBACK VISUAL: Emite mensagens via postMessage para atualizar a barra de progresso, gráficos no TF Visor
 *    e liberar o botão "Run Recommendation" na tela.
 * 
 * @param {Object} payload
 * @param {Array<Object>} payload.users - Lista de clientes e seus históricos vindos do Banco de Dados
 */
async function trainModel({ users }) {
    console.log('[Worker] Iniciando treinamento com clientes da padaria:', users.length);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 20 } });

    // 1. Obter os produtos de padaria exclusivamente da API do banco de dados vetorial
    let products = [];
    try {
        const prodRes = await fetch(`${API_BASE_URL}/products`);
        if (prodRes.ok) {
            products = await prodRes.json();
            console.log(`[Worker] ${products.length} produtos obtidos da API do Banco Vetorial.`);
        } else {
            throw new Error(`Falha na API: ${prodRes.status}`);
        }
    } catch (err) {
        console.error('[Worker] Erro crítico ao obter produtos do banco vetorial:', err);
        return;
    }

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 40 } });

    // 2. Gerar o contexto e os vetores numéricos de cada produto da padaria
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

    // 3. Persistir os embeddings gerados no Banco de Dados Vetorial SQLite via API!
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
        console.log(`[Worker] ${itemsToSave.length} embeddings de produtos salvos no Banco Vetorial!`);
    } catch (err) {
        console.warn('[Worker] Aviso ao persistir embeddings no banco:', err.message);
    }

    // 4. Treinar a rede neural
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 60 } });
    const trainData = createTrainingData(context);
    _model = await configureNeuralNetAndTrain(trainData);

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
}

async function recommend(user, ctx) {
    if (!_model) return;

    const context = _globalCtx;

    // Converte o usuário selecionado no vetor de features
    const userVector = encodeUser(user, context).dataSync();

    // =========================================================================
    // ETAPA 1: BUSCA NO BANCO DE DADOS VETORIAL (Vector Database Retrieval)
    // Busca no SQLite via SQL cosine_similarity os produtos de padaria mais próximos
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
        console.warn('[Worker] Busca vetorial na API não alcançada, usando lista local:', err);
    }

    // Mapeia os produtos a serem ranqueados pela rede neural
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
    // A rede neural calcula a pontuação final (0.00 a 1.00) para cada candidato
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
            score: scores[index],
            vectorSimilarity: item.vectorSimilarity !== undefined ? item.vectorSimilarity : null
        };
    });

    const sortedItems = recommendations.sort((a, b) => b.score - a.score);

    console.log('[Worker] Recomendações prontas para cliente:', user.name);

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
