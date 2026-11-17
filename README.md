# 🥐 Bakery Product Recommendation System com Banco de Dados Vetorial & Rede Neural

Sistema de recomendação de produtos de padaria utilizando **Redes Neurais com TensorFlow.js** combinado com **Banco de Dados Vetorial (Vector Database)** para busca por similaridade de cosseno e persistência completa em banco de dados.

O projeto utiliza o conjunto de dados real do arquivo **`data/bakery_sales_dataset.csv`**, que foi importado e estruturado diretamente dentro do **Banco de Dados Vetorial SQLite (`database/data.db`)**. A aplicação e o modelo realizam **100% das operações e consultas diretamente no Banco de Dados**, sem ler o arquivo `.csv` em tempo de execução.

---

## 🧠 Arquitetura: Recuperação Vetorial + Re-ranking Neural (Two-Stage Recommendation)

```
[ Cliente Selecionado da Padaria ]
          │
          ▼
   1. Vetor do Cliente (Embedding Numérico)
   (Idade normalizada + Histórico de Compras ponderado)
          │
          ▼
┌────────────────────────────────────────────────────────┐
│  ETAPA 1: Busca no Banco Vetorial (Vector Retrieval)   │
│  - Executa busca SQL por similaridade de cosseno:      │
│    ORDER BY cosine_similarity(embedding, ?) DESC      │
│  - Filtra rapidamente os produtos de padaria mais      │
│    promissores para aquele perfil                      │
└─────────────────────────┬──────────────────────────────┘
                          │ (Candidatos com score vetorial)
                          ▼
┌────────────────────────────────────────────────────────┐
│  ETAPA 2: Re-ranking com Rede Neural (TensorFlow.js)   │
│  - Combina [Vetor Cliente + Vetor Produto]             │
│  - Modelo Sequential (128 -> 64 -> 32 -> 1 Sigmoid)    │
│  - Calcula a probabilidade exata de compra             │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
     [ Recomendações Ordenadas com IA Score e Vetor % ]
```

---

## 💾 Banco de Dados Vetorial Integrado

1. **Dados Armazenados no Banco:**
   - **Tabela `products`:** 35 produtos distintos de padaria (Pastry, Chocolate Cake, Paav, Nankhatai, Coffee Powder, Bread, Cheese, etc.) com preço médio histórico, categoria, tipo de armazenamento (`storage`) e a coluna de vetor `embedding`.
   - **Tabela `users`:** Clientes reais com faixas etárias variadas (16 a 59 anos), segmentos (Student, Young Professional, Family) e seus perfis.
   - **Tabela `purchases`:** Histórico de compras relacionais persistido em disco.
   
2. **Capacidade Vetorial:**
   - Função SQL registrada: `cosine_similarity(vecA, vecB)`.
   - Os embeddings calculados pelo modelo após o pré-processamento das features da padaria são persistidos na coluna vetorial do banco de dados via `POST /api/products/embeddings`.
   - As recomendações realizam consulta vetorial via `POST /api/recommendations/vector-search`.

3. **Totalmente Desacoplado do CSV em Runtime:**
   - O arquivo CSV é utilizado apenas no seed inicial do banco de dados.
   - Todo o frontend, os controllers, serviços e o Web Worker consomem exclusivamente as APIs REST do banco de dados vetorial (`/api/products`, `/api/users`, etc.).

---

## 🚀 Como Executar

### 1. Testar os Endpoints e o Banco Vetorial
```bash
npm test
```

### 2. Iniciar a Aplicação
```bash
npm start
```

Abra o navegador em:
- **`http://localhost:3000`** (ou `http://localhost:3002`)

---

## 🔌 Endpoints da API REST

| Método | Rota | Descrição |
|---|---|---|
| `GET` | `/api/status` | Retorna o status do banco, total de produtos de padaria e vetores salvos |
| `GET` | `/api/products` | Lista os 35 produtos de padaria do banco |
| `POST` | `/api/products/embeddings` | Salva os vetores calculados pelo modelo na coluna vetorial |
| `POST` | `/api/recommendations/vector-search` | Executa busca vetorial SQL por similaridade de cosseno |
| `GET` | `/api/users` | Lista clientes e seus históricos de compras do banco |
| `POST` | `/api/users/:id/purchases` | Registra uma nova compra no banco relacional |
| `DELETE` | `/api/users/:id/purchases/:productId` | Remove uma compra do banco relacional |
| `POST` | `/api/database/reset` | Restaura o banco de dados com os dados da padaria |
