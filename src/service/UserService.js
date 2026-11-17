import { API_BASE_URL } from '../config.js';

export class UserService {
    #storageKey = 'ew-academy-users';
    #cachedUsers = [];

    async getDefaultUsers() {
        try {
            const response = await fetch(`${API_BASE_URL}/users`);
            if (response.ok) {
                const users = await response.json();
                this.#cachedUsers = users;
                this.#setStorage(users);
                return users;
            }
        } catch (err) {
            console.warn('[UserService] API indisponível, usando fallback local /data/users.json:', err.message);
        }

        const fallback = await fetch('/data/users.json');
        const users = await fallback.json();
        this.#cachedUsers = users;
        this.#setStorage(users);
        return users;
    }

    async getUsers() {
        try {
            const response = await fetch(`${API_BASE_URL}/users`);
            if (response.ok) {
                const users = await response.json();
                this.#cachedUsers = users;
                this.#setStorage(users);
                return users;
            }
        } catch (err) {
            console.warn('[UserService] Erro ao buscar usuários da API, usando armazenamento em cache:', err.message);
        }

        return this.#getStorage();
    }

    async getUserById(userId) {
        try {
            const response = await fetch(`${API_BASE_URL}/users/${userId}`);
            if (response.ok) {
                return await response.json();
            }
        } catch (err) {
            console.warn(`[UserService] Erro ao buscar usuário ${userId} da API:`, err.message);
        }

        const users = await this.getUsers();
        return users.find(u => Number(u.id) === Number(userId));
    }

    async updateUser(user) {
        // Obter o estado atual do usuário para detectar se foi adição ou remoção de compra
        const currentInDb = await this.getUserById(user.id);
        const oldPurchases = currentInDb?.purchases || [];
        const newPurchases = user.purchases || [];

        try {
            // Se adicionou uma compra nova
            if (newPurchases.length > oldPurchases.length) {
                const latestAdded = newPurchases[newPurchases.length - 1];
                if (latestAdded && latestAdded.id) {
                    await fetch(`${API_BASE_URL}/users/${user.id}/purchases`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ productId: latestAdded.id })
                    });
                }
            } 
            // Se removeu uma compra
            else if (newPurchases.length < oldPurchases.length) {
                // Identificar qual produto foi removido
                const oldIds = oldPurchases.map(p => p.id);
                const newIds = newPurchases.map(p => p.id);
                const removedId = oldIds.find(id => {
                    const oldCount = oldIds.filter(x => x === id).length;
                    const newCount = newIds.filter(x => x === id).length;
                    return oldCount > newCount;
                });

                if (removedId) {
                    await fetch(`${API_BASE_URL}/users/${user.id}/purchases/${removedId}`, {
                        method: 'DELETE'
                    });
                }
            }
        } catch (err) {
            console.error('[UserService] Erro ao sincronizar compra com o banco de dados:', err);
        }

        // Atualizar cache local
        const users = this.#getStorage();
        const userIndex = users.findIndex(u => Number(u.id) === Number(user.id));
        if (userIndex !== -1) {
            users[userIndex] = { ...users[userIndex], ...user };
        } else {
            users.push(user);
        }
        this.#setStorage(users);

        return user;
    }

    async addUser(user) {
        try {
            await fetch(`${API_BASE_URL}/users`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    id: user.id,
                    name: user.name,
                    age: user.age
                })
            });
        } catch (err) {
            console.warn('[UserService] Não foi possível salvar novo usuário no banco de dados:', err.message);
        }

        const users = this.#getStorage();
        this.#setStorage([user, ...users.filter(u => u.id !== user.id)]);
    }

    #getStorage() {
        const data = sessionStorage.getItem(this.#storageKey);
        return data ? JSON.parse(data) : this.#cachedUsers;
    }

    #setStorage(data) {
        sessionStorage.setItem(this.#storageKey, JSON.stringify(data));
    }
}
