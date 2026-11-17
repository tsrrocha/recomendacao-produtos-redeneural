-- ============================================================================
-- Esquema para PostgreSQL com extensão pgvector - Bakery Sales Dataset
-- Use este script caso queira rodar no Supabase, Neon ou PostgreSQL via Docker:
-- docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
-- ============================================================================

-- 1. Habilita a extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabela de Clientes da Padaria
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT NOT NULL,
    code VARCHAR(50),
    segment VARCHAR(100)
);

-- 3. Tabela de Produtos de Padaria com Coluna de Vetor (12 dimensões)
-- O vetor é composto por:
-- [preço_normalizado * peso, idade_média_normalizada * peso, 7 categorias one-hot, 3 storage one-hot] = 12 dims
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    storage VARCHAR(50) NOT NULL,
    color VARCHAR(50) NOT NULL,
    embedding vector(12) -- Coluna vetorial do pgvector!
);

-- 4. Tabela de Compras (Relacionamento N:N entre Clientes e Produtos de Padaria)
CREATE TABLE IF NOT EXISTS purchases (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id INT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Criação do Índice Vetorial HNSW para buscas por similaridade de cosseno
CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx 
ON products 
USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- EXEMPLO DE CONSULTA VETORIAL NO POSTGRESQL (pgvector):
-- Encontrar os 5 produtos de padaria mais similares ao perfil do cliente:
--
-- SELECT 
--     id, 
--     name, 
--     category, 
--     price, 
--     storage,
--     1 - (embedding <=> '[0.15, 0.08, 0.4, 0, 0, 0, 0, 0, 0, 0.3, 0, 0]') AS similarity
-- FROM products
-- WHERE embedding IS NOT NULL
-- ORDER BY embedding <=> '[0.15, 0.08, 0.4, 0, 0, 0, 0, 0, 0, 0.3, 0, 0]'
-- LIMIT 5;
-- ============================================================================
