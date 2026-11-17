import fs from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CSV_PATH = resolve(__dirname, '../data/bakery_sales_dataset.csv');

/**
 * Lê e processa o dataset de padaria do CSV de forma síncrona para popular o banco de dados.
 * @param {number} maxCustomers Quantidade de clientes mais representativos a importar
 */
export function parseBakeryCsvSync(maxCustomers = 35) {
    if (!fs.existsSync(CSV_PATH)) {
        throw new Error(`Arquivo de dados da padaria não encontrado: ${CSV_PATH}`);
    }

    const content = fs.readFileSync(CSV_PATH, 'utf-8');
    const lines = content.split(/\r?\n/);

    const productsMap = new Map();
    const customersMap = new Map();

    // Linha 0 é o cabeçalho
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        if (!line || !line.trim()) continue;

        const cols = line.split(',');
        if (cols.length < 20) continue;

        const custCode = cols[1]?.trim();
        const age = parseInt(cols[2], 10) || 30;
        const segment = cols[4]?.trim() || 'General';
        const productName = cols[18]?.trim();
        const category = cols[19]?.trim();
        const price = parseFloat(cols[21]) || parseFloat(cols[24]) || 15.0;
        const storage = cols[30]?.trim() || 'Room Temperature';

        if (!productName || !category) continue;

        // Agrupa produto para extrair os 35 produtos únicos e seus preços médios
        if (!productsMap.has(productName)) {
            productsMap.set(productName, {
                id: productsMap.size + 1,
                name: productName,
                category,
                priceSum: price,
                count: 1,
                storage
            });
        } else {
            const p = productsMap.get(productName);
            p.priceSum += price;
            p.count++;
        }

        // Agrupa cliente e suas compras
        if (!customersMap.has(custCode)) {
            customersMap.set(custCode, {
                code: custCode,
                age,
                segment,
                purchases: [productName]
            });
        } else {
            customersMap.get(custCode).purchases.push(productName);
        }
    }

    // Formata a lista definitiva dos produtos de padaria
    const products = Array.from(productsMap.values()).map(p => ({
        id: p.id,
        name: p.name,
        category: p.category,
        price: Number((p.priceSum / p.count).toFixed(2)),
        storage: p.storage,
        color: p.storage
    }));

    const productByName = new Map(products.map(p => [p.name, p]));

    // Seleciona clientes mais ativos com históricos de compras ricos
    const sortedCusts = Array.from(customersMap.values())
        .filter(c => c.purchases.length >= 10)
        .sort((a, b) => b.purchases.length - a.purchases.length)
        .slice(0, maxCustomers);

    const users = sortedCusts.map((c, index) => {
        // Pega compras distintas representativas do cliente (até 8 produtos distintos)
        const uniqueProductNames = [...new Set(c.purchases)].slice(0, 8);
        const userPurchases = uniqueProductNames
            .map(pName => productByName.get(pName))
            .filter(Boolean);

        return {
            id: index + 1,
            code: c.code,
            name: `${c.code} (${c.age}a - ${c.segment})`,
            age: c.age,
            segment: c.segment,
            purchases: userPurchases
        };
    });

    return { products, users };
}
