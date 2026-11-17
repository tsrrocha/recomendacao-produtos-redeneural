import { API_BASE_URL } from '../config.js';

export class ProductService {
    #cachedProducts = null;

    async getProducts(forceRefresh = false) {
        if (this.#cachedProducts && !forceRefresh) {
            return this.#cachedProducts;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/products`);
            if (response.ok) {
                const data = await response.json();
                this.#cachedProducts = data;
                return data;
            }
            throw new Error(`Falha ao obter produtos da API: ${response.status}`);
        } catch (err) {
            console.warn('[ProductService] Aviso: API não alcançada, usando fallback local /data/products.json:', err.message);
            const fallbackResponse = await fetch('/data/products.json');
            this.#cachedProducts = await fallbackResponse.json();
            return this.#cachedProducts;
        }
    }

    async getProductById(id) {
        const products = await this.getProducts();
        return products.find(product => product.id === Number(id));
    }

    async getProductsByIds(ids) {
        const products = await this.getProducts();
        return products.filter(product => ids.includes(product.id));
    }

    // Busca produtos similares usando o banco de dados vetorial
    async searchSimilar(userVector, limit = 10) {
        try {
            const response = await fetch(`${API_BASE_URL}/recommendations/vector-search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ userVector, limit })
            });

            if (response.ok) {
                const result = await response.json();
                return result.candidates || [];
            }
        } catch (err) {
            console.error('[ProductService] Erro na busca vetorial na API:', err);
        }
        return [];
    }

    // Persiste os embeddings calculados na coluna vetorial do banco de dados
    async saveEmbeddings(items) {
        try {
            const response = await fetch(`${API_BASE_URL}/products/embeddings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ items })
            });
            return await response.json();
        } catch (err) {
            console.error('[ProductService] Erro ao salvar embeddings na API:', err);
            return null;
        }
    }
}
