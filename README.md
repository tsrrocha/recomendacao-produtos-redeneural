# E-commerce Recommendation System com Banco de Dados Vetorial & Rede Neural

Sistema de recomendação de produtos em e-commerce utilizando **Redes Neurais com TensorFlow.js** combinado com **Banco de Dados Vetorial (Vector Database)** para busca por similaridade e persistência de dados.

---

## 🧠 Arquitetura: Recuperação Vetorial + Re-ranking Neural (Two-Stage Recommendation)

Em sistemas de recomendação modernos de grande escala (como Netflix, Amazon, Spotify), o catálogo possui milhares ou milhões de produtos. Avaliar uma rede neural pesada em todos os produtos seria lento e computacionalmente inviável.

Por isso, utilizamos a arquitetura em duas etapas:

```
[ Usuário Selecionado ]
          │
          ▼
   1. Vetor do Usuário (Embedding de 14 Dimensões)
   (Idade normalizada + Histórico de Compras ponderado)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  ETAPA 1: Busca no Banco Vetorial (Vector Retrieval)   │
│  - Executa busca SQL por similaridade de cosseno:      │
│    ORDER BY cosine_similarity(embedding, ?) DESC      │
│  - Filtra rapidamente os produtos mais promissores     │
└─────────────────────────┬──────────────────────────────┘
                          │ (Candidatos com score vetorial)
                          ▼
┌────────────────────────────────────────────────────────┐
│  ETAPA 2: Re-ranking com Rede Neural (TensorFlow.js)   │
│  - Combina [Vetor Usuário + Vetor Produto]             │
│  - Modelo Sequential (128 -> 64 -> 32 -> 1 Sigmoid)    │
│  - Calcula a probabilidade exata de compra             │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
     [ Recomendações Ordenadas com IA Score e Vetor % ]
```

---

## 💾 Banco de Dados Vetorial Integrado

Anteriormente, o projeto lia dados de arquivos `.json` estáticos e salvava compras apenas na memória da sessão (`sessionStorage`).

Agora, o projeto possui um **Banco de Dados Relacional e Vetorial** completo:

1. **Persistência em Disco:**
   - Salvo em `database/data.db` via SQLite nativo do Node.js (`node:sqlite`).
   - Tabelas estruturadas: `products`, `users` e `purchases` com chaves estrangeiras.
   - Qualquer nova compra via botão **"Buy Now"** ou remoção de compra é persistida de verdade no banco.

2. **Capacidade Vetorial:**
   - Coluna `embedding` na tabela `products`.
   - Funções SQL registradas: `cosine_similarity(vecA, vecB)` e `l2_distance(vecA, vecB)`.
   - Os embeddings são gerados pelo modelo no pré-processamento e salvos automaticamente no banco de dados.

3. **Compatibilidade com PostgreSQL + pgvector:**
   - Foi disponibilizado o arquivo `database/schema_postgres.sql`.
   - Possui a sintaxe exata do PostgreSQL com a extensão `pgvector` (`CREATE EXTENSION vector;`, coluna `vector(14)`, índice `HNSW` e operador `<=>`), permitindo fácil migração para Supabase, Neon ou Docker.

---

## 🚀 Como Executar

### Pré-requisitos
- Node.js instalado (v18+)

### 1. Instalar as dependências
```bash
npm install
```

### 2. Iniciar o servidor
```bash
npm start
```

O servidor iniciará em `http://localhost:3001` (ou `http://localhost:3000`).

Abra o navegador no endereço exibido no terminal.

---

## 🔌 Endpoints da API REST

A API do banco de dados vetorial disponibiliza:

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/status` | Retorna o status do banco, total de produtos e vetores indexados |
| `GET` | `/api/products` | Lista todos os produtos cadastrados |
| `POST` | `/api/products/embeddings` | Salva os vetores (embeddings) calculados no banco |
| `POST` | `/api/recommendations/vector-search` | Executa a busca vetorial via SQL por similaridade de cosseno |
| `GET` | `/api/users` | Lista usuários e seus históricos de compras do banco |
| `POST` | `/api/users/:id/purchases` | Registra uma nova compra no banco relacional |
| `DELETE` | `/api/users/:id/purchases/:productId` | Remove uma compra do banco relacional |
| `POST` | `/api/database/reset` | Restaura os dados originais a partir dos JSONs |

---

## 📁 Estrutura de Arquivos

```
├── database/
│   ├── db.js                   # Módulo do banco SQLite com suporte vetorial
│   ├── data.db                 # Arquivo de dados persistido em disco
│   └── schema_postgres.sql     # Script de referência para PostgreSQL + pgvector
├── server.js                   # Servidor Express com API REST e arquivos estáticos
├── index.html                  # Interface gráfica web
├── style.css                   # Estilização
├── src/
│   ├── config.js               # Configuração da URL da API
│   ├── index.js                # Inicialização dos controllers e status do banco
│   ├── controller/             # Controladores MVC
│   ├── service/
│   │   ├── ProductService.js   # Serviço de produtos conectado à API vetorial
│   │   └── UserService.js      # Serviço de usuários com compras persistidas
│   ├── view/                   # Visualização e templates
│   └── workers/
│       └── modelTrainingWorker.js # Worker do TensorFlow.js integrado ao Banco Vetorial
└── data/                       # Arquivos JSON originais para seed inicial
```
