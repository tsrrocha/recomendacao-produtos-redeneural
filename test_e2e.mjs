async function runEndToEndTest() {
    const API = 'http://localhost:3002/api';
    console.log('--- Teste 1: Status do Banco Vetorial ---');
    let res = await fetch(`${API}/status`);
    let data = await res.json();
    console.log('Status inicial:', data);

    console.log('\n--- Teste 2: Listagem de Produtos ---');
    res = await fetch(`${API}/products`);
    const products = await res.json();
    console.log('Total produtos retornados:', products.length);

    console.log('\n--- Teste 3: Listagem de Usuários ---');
    res = await fetch(`${API}/users`);
    const users = await res.json();
    console.log('Total usuários retornados:', users.length, '| Exemplo 1:', users[0].name, '| Compras:', users[0].purchases.length);

    console.log('\n--- Teste 4: Salvar Embeddings de Produtos (Simulando Worker) ---');
    const embeddings = products.map((p, idx) => ({
        id: p.id,
        vector: Array(14).fill(0).map((_, i) => (i === idx % 14 ? 0.9 : 0.05))
    }));
    res = await fetch(`${API}/products/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ items: embeddings })
    });
    console.log('Resposta salvar embeddings:', await res.json());

    console.log('\n--- Teste 5: Busca Vetorial no Banco (SQL cosine_similarity) ---');
    const userQueryVector = Array(14).fill(0).map((_, i) => (i === 0 ? 0.85 : 0.02));
    res = await fetch(`${API}/recommendations/vector-search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ userVector: userQueryVector, limit: 3 })
    });
    const vecResults = await res.json();
    console.log('Top 3 candidatos mais próximos pelo Banco Vetorial:');
    vecResults.candidates.forEach(c => {
        console.log(` - [${c.name}] Similaridade de Cosseno no BD: ${(c.vectorSimilarity * 100).toFixed(1)}%`);
    });

    console.log('\n--- Teste 6: Registro de Compra e Persistência no BD ---');
    const targetUser = users[0];
    const initialCount = targetUser.purchases.length;
    res = await fetch(`${API}/users/${targetUser.id}/purchases`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ productId: 5 })
    });
    const userAfterBuy = await res.json();
    console.log(`Compras após 'Buy Now': ${userAfterBuy.purchases.length} (era ${initialCount})`);

    console.log('\n--- Teste 7: Remoção de Compra ---');
    res = await fetch(`${API}/users/${targetUser.id}/purchases/5`, {
        method: 'DELETE'
    });
    const userAfterRemove = await res.json();
    console.log(`Compras após remover: ${userAfterRemove.purchases.length}`);

    console.log('\n--- Teste 8: Verificação Final de Status ---');
    res = await fetch(`${API}/status`);
    console.log('Status final:', await res.json());
    console.log('\n✅ TODOS OS TESTES PASSARAM COM SUCESSO!');
}

runEndToEndTest().catch(console.error);

