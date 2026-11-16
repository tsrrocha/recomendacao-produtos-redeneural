import 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';
import { workerEvents } from '../events/constants.js';

console.log('Model training worker initialized');
let _globalCtx = {};
let _model = null

const WEIGHTS = {
    category: 0.4,
    color: 0.3,
    price: 0.2,
    age: 0.1
}

/*===================================================================================================|                                   SETUP DO DATASET PARA TREINAMENTO                                 ||  ==
        Explicação:
=======================================================================================================

Exemplo de Usuário:
{
    "id": 1,
    "name": "Ana Lima",
    "age": 25,
    "purchases": [
        {
            "id": 1,  "name": "Fones de Ouvido Sem Fio", "category": "eletrônicos", "price": 129.99, "color": "preto"
        }
    ]
}


Após a codificação, o modelo não nomes ou palavras. Ele vê um VETOR NUMÉRICO (todos normalizados entre 0-1)
Exemplo: [preco_normalizado, idade_normalizada, cat_one_hot, cor_one_hot]



=======================================================================================================
*/


const normalize = (value, min, max) => (value - min) / ((max - min) || 1)

function makeContext(products, users) {
    const ages = users.map(u => u.age)
    const prices = products.map(p => p.price)

    // Normalizando os dados
    // Coleta o mínimo e máximo da idade
    const minAge = Math.min(...ages)
    const maxAge = Math.max(...ages)
    // Coleta o mínimo e máximo preço
    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)
    // Normaliza as Cores
    const colors = [...new Set(products.map(p => p.color))]
    const categories = [...new Set(products.map(p => p.category))]

    const colorsIndex = Object.fromEntries(
        colors.map((color, index) => {
            return [color, index]
        })
    )

    const categoriesIndex = Object.fromEntries(
        categories.map((category, index) => {
            return [category, index]
        })
    )

    // Computar a média de idade dos comprados por produto 
    // (ajuda a personalizar)
    const midAge = (minAge + maxAge) / 2
    const ageSums = {}
    const ageCounts = {}

    // Iterar nos usuários
    users.forEach(user => {
        user.purchases.forEach(p => {
            ageSums[p.name] = (ageSums[p.name] || 0) + user.age;
            ageCounts[p.name] = (ageCounts[p.name] || 0) + 1;
        })
    });

    //
    const productAvgAgeNorm = Object.fromEntries(
        products.map(product => {
            const avg = ageCounts[product.name] ?
                ageSums[product.name] / ageCounts[product.name] :
                midAge

            return [product.name, normalize(avg, minAge, maxAge)]
        })
    )

    // Retornar o tensor
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
    }
}

const oneHotWeighted = (index, length, weight) =>
    tf.oneHot(index, length).cast('float32').mul(weight)

function encodeProduct(product, context) {
    // Normalizand o dado para ficar de 0 a 1
    // aplicar o peso na recomendação
    const price = tf.tensor1d([
        normalize(
            product.price,
            context.minPrice,
            context.maxPrice
        ) * WEIGHTS.price
    ])

    const age = tf.tensor1d([
        (
            context.productAvgAgeNorm[product.name] ?? 0.5
        ) * WEIGHTS.age
    ])

    const category = oneHotWeighted(
        context.categoriesIndex[product.category],
        context.numCategories,
        WEIGHTS.category
    )

    const color = oneHotWeighted(
        context.colorsIndex[product.color],
        context.numColors,
        WEIGHTS.color
    )

    return tf.concat1d(
        [price, age, category, color]
    )
}

function encodeUser(user, context) {
    if (user.purchases.length) {
        return tf.stack(
            user.purchases.map(
                product => encodeProduct(product, context)
            )
        )
            .mean(0)
            .reshape([
                1,
                context.dimentions
            ])
    }

    return tf.concat1d(
        [
            tf.zeros([1]), // O preço é ignorado
            tf.tensor1d([
                normalize(user.age, context.minAge, context.maxAge)
                * WEIGHTS.age
            ]),
            tf.zeros([context.numCategories]), // Categorias ignoradas
            tf.zeros([context.numColors]), // Cores ignoradas
        ]
    ).reshape([1, context.dimentions])
}

function createTrainingData(context) {
    const inputs = []
    const labels = []
    context.users
        .filter(u => u.purchases.length)
        .forEach(user => {
            const userVector = encodeUser(user, context).dataSync()
            context.products.forEach(product => {
                const productVector = encodeProduct(product, context).dataSync()

                const label = user.purchases.some(
                    purchase => purchase.name === product.name ? 1 : 0
                )

                // combinar user + product
                inputs.push([...userVector, ...productVector])
                labels.push(label)
            })
        })

    return {
        xs: tf.tensor2d(inputs),
        ys: tf.tensor2d(labels, [labels.length, 1]),
        inputDimention: context.dimentions * 2
        // Tamanho  = userVector + productVector
    }
}


/*===================================================================================================
        Configuração e Treinamento da Rede Neural
===================================================================================================*/
async function configureNeuralNetAndTrain(trainData) {
    const model = tf.sequential()

    // Camada de entrada 
    // inputShape: Número de features por exemplo de treino (trainData.inputDimention)
    // Exemplo: Se o vetor Usuario+Produto = 20 números, então inputDim = 20
    // units: 128 neurônios (muitos olhos para detectar padrões)
    // activation: 'relu' (mantém apenas sinais positivos, ajuda a aprender padrões não-lineares)
    model.add(
        tf.layers.dense({
            inputShape: [trainData.inputDimention],
            units: 128,
            activation: 'relu'
        })
    )

    // Camada oculta 1
    // - 64 neurônios: menos que a primeira camada (começa a comprimir a informação)
    model.add(
        tf.layers.dense({
            units: 64,
            activation: 'relu'
        })
    )

    // Camada oculta 2
    // - 32 neurônios: menos que a primeira camada (começa a comprimir a informação)
    model.add(
        tf.layers.dense({
            units: 32,
            activation: 'relu'
        })
    )

    // Camada de Saída
    // 1 neurônio porque vamos retornar apenas uma pontuação de recomendação
    // activation sigmoid: compirme o resultado para o intervalo 0 - 1.
    model.add(
        tf.layers.dense({ units: 1, activation: 'sigmoid' })
    )

    model.compile({
        optimizer: tf.train.adam(0.01),
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    })

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
    })

    return model
}


async function trainModel({ users }) {
    console.log('Training model with users:', users)

    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 50 } });

    // 
    const products = await (await fetch('/data/products.json')).json()

    const context = makeContext(products, users)
    context.productVectors = products.map(product => {
        return {
            name: product.name,
            meta: { ...product },
            vector: encodeProduct(product, context).dataSync()
        }
    })


    _globalCtx = context;

    const trainData = createTrainingData(context)

    _model = await configureNeuralNetAndTrain(trainData)




    //postMessage({
    //    type: workerEvents.trainingLog,
    //    epoch: 1,
    //    loss: 1,
    //    accuracy: 1
    //});

    //setTimeout(() => {
    postMessage({ type: workerEvents.progressUpdate, progress: { progress: 100 } });
    postMessage({ type: workerEvents.trainingComplete });
    //}, 1000);


}
function recommend(user, ctx) {
    if (!_model) return;

    const context = _globalCtx;

    // Converta o usuário fornecido no vetor de features codificadas
    // (preço ignorado, idade normalizada, categorias one-hot, cores one-hot)
    // Isso transforma as informações do usuário no mesmo formato numérico
    // que foi utilizado para treinar o modelo.
    const userVector = encodeUser(user, context).dataSync()

    // Em aplicações reais: 
    // Armazene todos os vetores de produtos em um banco de dados vetorial
    // (como PostgreSQL, Neo4j ou Pinecone)
    // Consulta: Encontre os 200 produtos mais próximos do vetor do usuário
    // Execute _model.predict() apenas nesses produtos

    // Crie pares de entrada: para cada produto, concatene o vetor de usuário
    // com o vetor codificado do produto.
    // Por quê? O modelo prevê o "score de compatibilidade" para cada par (usuário, produto)
    const inputs = context.productVectors.map(({ vector }) => {
        return [...userVector, ...vector]
    })

    // Converta todos esses pares (usuario, produto) em um único Tensor.
    // Formato: [numProdutos, inputDim]
    const inputTensor = tf.tensor2d(inputs)

    // Roda a rede neural treinada em todos os pares (usuário, produto) de uma vez
    // O resultado é uma pontuação para cada produto entre 0 e 1.
    // Quanto maior, maior a probabilidade do usuário querer aquele produto.
    const predictions = _model.predict(inputTensor)

    // Extraiaas pontuações para um Array JS normal
    const scores = predictions.dataSync()

    const recommendations = context.productVectors.map((item, index) => {
        return {
            ...item.meta,
            name: item.name,
            score: scores[index] // previsão do modelo para este produto
        }
    })

    const sortedItems = recommendations
        .sort((a, b) => b.score - a.score)


    //debugger
    console.log('will recommend for user:', user)
    // Envie a lista ordenada de produtos recomendados
    // para a thread principal (a UI pode exibi-los agora).
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

self.onmessage = e => {
    const { action, ...data } = e.data;
    if (handlers[action]) handlers[action](data);
};
