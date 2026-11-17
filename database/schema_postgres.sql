-- ============================================================================
-- Esquema para PostgreSQL com extensão pgvector
-- Use este script caso queira rodar no Supabase, Neon ou PostgreSQL via Docker:
-- docker run -p 5432:5432 -e POSTGRES_PASSWORD=postgres pgvector/pgvector:pg16
-- ============================================================================

-- 1. Habilita a extensão pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Tabela de Usuários
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    age INT NOT NULL
);

-- 3. Tabela de Produtos com coluna de VETOR (14 dimensões no nosso modelo)
-- O vetor é composto por:
-- [preço_normalizado * peso, idade_média_normalizada * peso, 4 categorias one-hot, 8 cores one-hot] = 14 dims
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price NUMERIC(10, 2) NOT NULL,
    color VARCHAR(50) NOT NULL,
    embedding vector(14) -- Coluna de vetor do pgvector!
);

-- 4. Tabela de Compras (Relacionamento N:N entre Usuários e Produtos)
CREATE TABLE IF NOT EXISTS purchases (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id INT NOT NULL REFERENCES products(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Criação do Índice Vetorial HNSW para buscas ultra-rápidas (Cosine Distance)
CREATE INDEX IF NOT EXISTS products_embedding_hnsw_idx 
ON products 
USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- EXEMPLO DE CONSULTA VETORIAL NO POSTGRESQL (pgvector):
-- Encontrar os 5 produtos mais próximos do vetor do usuário:
--
-- SELECT 
--     id, 
--     name, 
--     category, 
--     price, 
--     color,
--     1 - (embedding <=> '[0.05, 0.08, 0.4, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0]') AS similarity
-- FROM products
-- WHERE embedding IS NOT NULL
-- ORDER BY embedding <=> '[0.05, 0.08, 0.4, 0, 0, 0, 0.3, 0, 0, 0, 0, 0, 0, 0]'
-- LIMIT 5;
--
-- O operador <=> calcula a Distância de Cosseno (Cosine Distance).
-- 1 - (distância) resulta na similaridade de cosseno (de 0 a 1).
-- ============================================================================

