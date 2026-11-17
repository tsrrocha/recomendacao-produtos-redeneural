import { DatabaseSync } from 'node:sqlite';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseBakeryCsvSync } from './seed_bakery.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const DB_PATH = resolve(__dirname, 'data.db');

class VectorDatabase {
    #db;

    constructor() {
        this.init();
    }

    init() {
        // Inicializa o banco de dados SQLite persistente em disco
        this.#db = new DatabaseSync(DB_PATH);

        // Registra a função SQL de similaridade de cosseno
        // similaridade = (A . B) / (||A|| * ||B||)
        this.#db.function('cosine_similarity', (vecAJson, vecBJson) => {
            if (!vecAJson || !vecBJson) return 0;
            try {
                const vecA = typeof vecAJson === 'string' ? JSON.parse(vecAJson) : vecAJson;
                const vecB = typeof vecBJson === 'string' ? JSON.parse(vecBJson) : vecBJson;

                if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
                    return 0;
                }

                let dotProduct = 0;
                let normA = 0;
                let normB = 0;

                for (let i = 0; i < vecA.length; i++) {
                    const a = vecA[i];
                    const b = vecB[i];
                    dotProduct += a * b;
                    normA += a * a;
                    normB += b * b;
                }

                const denominator = Math.sqrt(normA) * Math.sqrt(normB);
                return denominator === 0 ? 0 : dotProduct / denominator;
            } catch (err) {
                return 0;
            }
        });

        // Registra a distância euclidiana (L2) no SQL
        this.#db.function('l2_distance', (vecAJson, vecBJson) => {
            if (!vecAJson || !vecBJson) return 999999;
            try {
                const vecA = typeof vecAJson === 'string' ? JSON.parse(vecAJson) : vecAJson;
                const vecB = typeof vecBJson === 'string' ? JSON.parse(vecBJson) : vecBJson;

                if (!Array.isArray(vecA) || !Array.isArray(vecB) || vecA.length !== vecB.length) {
                    return 999999;
                }

                let sumSq = 0;
                for (let i = 0; i < vecA.length; i++) {
                    const diff = vecA[i] - vecB[i];
                    sumSq += diff * diff;
                }
                return Math.sqrt(sumSq);
            } catch (err) {
                return 999999;
            }
        });

        // Cria as tabelas necessárias
        this.createTables();

        // Se o banco estiver vazio ou com o schema antigo, popula com o dataset da padaria
        this.seedIfEmpty();
    }

    createTables() {
        this.#db.exec(`
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                storage TEXT NOT NULL DEFAULT 'Room Temperature',
                color TEXT NOT NULL DEFAULT 'Room Temperature',
                embedding TEXT -- Vetor numérico serializado em JSON
            );
        `);

        // Garante que a coluna storage existe caso a tabela tenha sido criada na versão anterior
        try {
            this.#db.exec(`ALTER TABLE products ADD COLUMN storage TEXT NOT NULL DEFAULT 'Room Temperature';`);
        } catch (e) {
            // Coluna já existe
        }

        this.#db.exec(`
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                code TEXT,
                segment TEXT
            );
        `);

        try {
            this.#db.exec(`ALTER TABLE users ADD COLUMN code TEXT;`);
        } catch (e) {}

        try {
            this.#db.exec(`ALTER TABLE users ADD COLUMN segment TEXT;`);
        } catch (e) {}

        this.#db.exec(`
            CREATE TABLE IF NOT EXISTS purchases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY (product_id) REFERENCES products(id) ON DELETE CASCADE
            );
        `);
    }

    seedIfEmpty() {
        const productCount = this.#db.prepare('SELECT count(*) as count FROM products').get().count;
        // Se estiver com menos de 35 produtos, precisa atualizar para o dataset da padaria
        if (productCount < 35) {
            this.seed();
        }
    }

    seed() {
        console.log('[Database] Populando banco de dados com os dados do bakery_sales_dataset.csv...');
        const { products, users } = parseBakeryCsvSync(35);

        // Limpa tabelas existentes
        this.#db.exec(`
            DELETE FROM purchases;
            DELETE FROM users;
            DELETE FROM products;
        `);

        // Insere os 35 produtos da padaria
        const insertProduct = this.#db.prepare(`
            INSERT INTO products (id, name, category, price, storage, color, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        `);

        for (const p of products) {
            insertProduct.run(p.id, p.name, p.category, p.price, p.storage, p.color, null);
        }

        // Insere os clientes da padaria e seus históricos de compras
        const insertUser = this.#db.prepare(`
            INSERT INTO users (id, name, age, code, segment)
            VALUES (?, ?, ?, ?, ?)
        `);

        const insertPurchase = this.#db.prepare(`
            INSERT INTO purchases (user_id, product_id)
            VALUES (?, ?)
        `);

        for (const u of users) {
            insertUser.run(u.id, u.name, u.age, u.code, u.segment);
            if (Array.isArray(u.purchases)) {
                for (const pur of u.purchases) {
                    insertPurchase.run(u.id, pur.id);
                }
            }
        }

        console.log(`[Database] Seed concluído: ${products.length} produtos de padaria e ${users.length} clientes carregados no banco!`);
    }

    // ==================== OPERAÇÕES DE PRODUTOS ====================

    getProducts() {
        const stmt = this.#db.prepare('SELECT id, name, category, price, storage, color, embedding FROM products ORDER BY id');
        const rows = stmt.all();
        return rows.map(r => ({
            ...r,
            embedding: r.embedding ? JSON.parse(r.embedding) : null
        }));
    }

    getProductById(id) {
        const stmt = this.#db.prepare('SELECT id, name, category, price, storage, color, embedding FROM products WHERE id = ?');
        const row = stmt.get(Number(id));
        if (!row) return null;
        return {
            ...row,
            embedding: row.embedding ? JSON.parse(row.embedding) : null
        };
    }

    // Atualiza o embedding gerado pelo modelo no banco de dados vetorial
    updateProductEmbedding(id, embeddingArray) {
        const embeddingJson = JSON.stringify(Array.from(embeddingArray));
        const stmt = this.#db.prepare('UPDATE products SET embedding = ? WHERE id = ?');
        stmt.run(embeddingJson, Number(id));
    }

    // Atualiza embeddings em lote
    updateAllProductEmbeddings(items) {
        const stmt = this.#db.prepare('UPDATE products SET embedding = ? WHERE id = ?');
        for (const item of items) {
            const vec = Array.from(item.vector);
            stmt.run(JSON.stringify(vec), Number(item.id));
        }
    }

    // ==================== BUSCA VETORIAL (VECTOR SIMILARITY SEARCH) ====================
    // Executa a busca no banco vetorial ordenando por similaridade de cosseno via SQL
    searchSimilarProducts(userVector, limit = 10) {
        const queryVectorJson = JSON.stringify(Array.from(userVector));

        const stmt = this.#db.prepare(`
            SELECT 
                p.id, 
                p.name, 
                p.category, 
                p.price, 
                p.storage,
                p.color,
                p.embedding,
                cosine_similarity(p.embedding, ?) AS vector_similarity
            FROM products p
            WHERE p.embedding IS NOT NULL
            ORDER BY vector_similarity DESC
            LIMIT ?
        `);

        const rows = stmt.all(queryVectorJson, Number(limit));

        return rows.map(r => ({
            id: r.id,
            name: r.name,
            category: r.category,
            price: r.price,
            storage: r.storage,
            color: r.color,
            vectorSimilarity: Number(r.vector_similarity.toFixed(4)),
            embedding: r.embedding ? JSON.parse(r.embedding) : null
        }));
    }

    // ==================== OPERAÇÕES DE USUÁRIOS E COMPRAS ====================

    getUsers() {
        const usersStmt = this.#db.prepare('SELECT id, name, age, code, segment FROM users ORDER BY id');
        const users = usersStmt.all();

        const purchasesStmt = this.#db.prepare(`
            SELECT p.id, p.name, p.category, p.price, p.storage, p.color
            FROM purchases pur
            JOIN products p ON pur.product_id = p.id
            WHERE pur.user_id = ?
            ORDER BY pur.id
        `);

        return users.map(user => {
            const purchases = purchasesStmt.all(user.id);
            return {
                ...user,
                purchases
            };
        });
    }

    getUserById(id) {
        const userStmt = this.#db.prepare('SELECT id, name, age, code, segment FROM users WHERE id = ?');
        const user = userStmt.get(Number(id));
        if (!user) return null;

        const purchasesStmt = this.#db.prepare(`
            SELECT p.id, p.name, p.category, p.price, p.storage, p.color
            FROM purchases pur
            JOIN products p ON pur.product_id = p.id
            WHERE pur.user_id = ?
            ORDER BY pur.id
        `);

        const purchases = purchasesStmt.all(user.id);
        return {
            ...user,
            purchases
        };
    }

    addUser({ id, name, age, segment = 'New Customer' }) {
        const existing = this.getUserById(id);
        if (existing) {
            const updateStmt = this.#db.prepare('UPDATE users SET name = ?, age = ?, segment = ? WHERE id = ?');
            updateStmt.run(name, Number(age), segment, Number(id));
            return this.getUserById(id);
        }

        const insertStmt = this.#db.prepare('INSERT INTO users (id, name, age, segment) VALUES (?, ?, ?, ?)');
        insertStmt.run(Number(id), name, Number(age), segment);
        return this.getUserById(id);
    }

    addPurchase(userId, productId) {
        const insertStmt = this.#db.prepare('INSERT INTO purchases (user_id, product_id) VALUES (?, ?)');
        insertStmt.run(Number(userId), Number(productId));
        return this.getUserById(userId);
    }

    removePurchase(userId, productId) {
        const findStmt = this.#db.prepare(`
            SELECT id FROM purchases 
            WHERE user_id = ? AND product_id = ? 
            ORDER BY id DESC LIMIT 1
        `);
        const item = findStmt.get(Number(userId), Number(productId));
        if (item) {
            const deleteStmt = this.#db.prepare('DELETE FROM purchases WHERE id = ?');
            deleteStmt.run(item.id);
        }
        return this.getUserById(userId);
    }

    getStats() {
        const productCount = this.#db.prepare('SELECT count(*) as count FROM products').get().count;
        const vectorCount = this.#db.prepare('SELECT count(*) as count FROM products WHERE embedding IS NOT NULL').get().count;
        const userCount = this.#db.prepare('SELECT count(*) as count FROM users').get().count;
        const purchaseCount = this.#db.prepare('SELECT count(*) as count FROM purchases').get().count;

        return {
            engine: 'SQLite Vector DB (Built-in Node.js node:sqlite)',
            dataset: 'Bakery Sales Dataset (Padaria)',
            databasePath: DB_PATH,
            totalProducts: productCount,
            productsWithVectors: vectorCount,
            totalUsers: userCount,
            totalPurchases: purchaseCount
        };
    }
}

export const db = new VectorDatabase();
