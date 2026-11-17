import express from 'express';
import cors from 'cors';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { db } from './database/db.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Servir arquivos estáticos do frontend (index.html, style.css, src/, etc.)
app.use(express.static(__dirname));

// ==================== ENDPOINTS DA API DO BANCO VETORIAL ====================

// Status do banco de dados
app.get('/api/status', (req, res) => {
    try {
        const stats = db.getStats();
        res.json({ success: true, ...stats });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

// Listar todos os produtos
app.get('/api/products', (req, res) => {
    try {
        const products = db.getProducts();
        res.json(products);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Buscar produto por ID
app.get('/api/products/:id', (req, res) => {
    try {
        const product = db.getProductById(req.params.id);
        if (!product) return res.status(404).json({ error: 'Produto não encontrado' });
        res.json(product);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Salvar / atualizar embeddings de produtos gerados pelo modelo no banco de dados vetorial
app.post('/api/products/embeddings', (req, res) => {
    try {
        const { items } = req.body;
        if (!Array.isArray(items)) {
            return res.status(400).json({ error: 'items deve ser um array com { id, vector }' });
        }

        db.updateAllProductEmbeddings(items);
        console.log(`[Vector DB] ${items.length} embeddings de produtos salvos no banco de dados!`);
        res.json({ success: true, message: `${items.length} embeddings atualizados com sucesso` });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Busca vetorial: encontra os produtos mais similares ao vetor do usuário diretamente via SQL
app.post('/api/recommendations/vector-search', (req, res) => {
    try {
        const { userVector, limit = 10 } = req.body;
        if (!userVector || !Array.isArray(userVector)) {
            return res.status(400).json({ error: 'userVector é obrigatório e deve ser um array numérico' });
        }

        const candidates = db.searchSimilarProducts(userVector, limit);
        console.log(`[Vector Search] Busca vetorial executada! ${candidates.length} candidatos encontrados.`);
        res.json({ success: true, candidates });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Listar todos os usuários com compras
app.get('/api/users', (req, res) => {
    try {
        const users = db.getUsers();
        res.json(users);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Buscar usuário por ID
app.get('/api/users/:id', (req, res) => {
    try {
        const user = db.getUserById(req.params.id);
        if (!user) return res.status(404).json({ error: 'Usuário não encontrado' });
        res.json(user);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Criar / atualizar usuário
app.post('/api/users', (req, res) => {
    try {
        const { id, name, age } = req.body;
        if (!id || !name || !age) {
            return res.status(400).json({ error: 'id, name e age são obrigatórios' });
        }
        const user = db.addUser({ id, name, age });
        res.json(user);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Adicionar compra para um usuário
app.post('/api/users/:id/purchases', (req, res) => {
    try {
        const userId = req.params.id;
        const { productId } = req.body;
        if (!productId) {
            return res.status(400).json({ error: 'productId é obrigatório' });
        }
        const updatedUser = db.addPurchase(userId, productId);
        console.log(`[Database] Compra registrada no banco: Usuário ${userId} comprou Produto ${productId}`);
        res.json(updatedUser);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Remover compra de um usuário
app.delete('/api/users/:id/purchases/:productId', (req, res) => {
    try {
        const { id: userId, productId } = req.params;
        const updatedUser = db.removePurchase(userId, productId);
        console.log(`[Database] Compra removida no banco: Usuário ${userId}, Produto ${productId}`);
        res.json(updatedUser);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// Resetar o banco para o estado inicial
app.post('/api/database/reset', (req, res) => {
    try {
        db.seed();
        res.json({ success: true, message: 'Banco de dados restaurado com sucesso!' });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

const DEFAULT_PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3001;

const server = app.listen(DEFAULT_PORT, () => {
    console.log(`\n==================================================`);
    console.log(`🚀 Servidor rodando em: http://localhost:${DEFAULT_PORT}`);
    console.log(`💾 Banco de Dados Vetorial: SQLite Local (node:sqlite)`);
    console.log(`==================================================\n`);
}).on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
        const fallbackPort = DEFAULT_PORT + 1;
        console.warn(`[Aviso] Porta ${DEFAULT_PORT} em uso. Tentando porta ${fallbackPort}...`);
        app.listen(fallbackPort, () => {
            console.log(`\n==================================================`);
            console.log(`🚀 Servidor rodando na porta alternativa: http://localhost:${fallbackPort}`);
            console.log(`💾 Banco de Dados Vetorial: SQLite Local (node:sqlite)`);
            console.log(`==================================================\n`);
        });
    } else {
        console.error('Erro no servidor:', err);
    }
});
