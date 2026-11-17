import { View } from './View.js';

export class ProductView extends View {
    // DOM elements
    #productList = document.querySelector('#productList');

    #buttons;
    // Templates and callbacks
    #productTemplate;
    #onBuyProduct;

    constructor() {
        super();
        this.init();
    }

    async init() {
        this.#productTemplate = await this.loadTemplate('./src/view/templates/product-card.html');
    }

    onUserSelected(user) {
        // Enable buttons if a user is selected, otherwise disable them
        this.setButtonsState(user && user.id ? false : true);
    }

    registerBuyProductCallback(callback) {
        this.#onBuyProduct = callback;
    }

    render(products, disableButtons = true) {
        if (!this.#productTemplate) return;
        const html = products.map(product => {
            let scoreBadge = '';
            if (product.score !== undefined) {
                const pct = Math.round(product.score * 100);
                const vecSim = product.vectorSimilarity !== null && product.vectorSimilarity !== undefined
                    ? Math.round(product.vectorSimilarity * 100)
                    : null;

                scoreBadge = `
                    <div class="mt-2 pt-2 border-top d-flex flex-wrap gap-1">
                        <span class="badge bg-success" title="Previsão da Rede Neural">
                            <i class="bi bi-cpu"></i> IA Score: ${pct}%
                        </span>
                        ${vecSim !== null ? `
                        <span class="badge bg-info text-dark" title="Similaridade Vetorial no Banco de Dados">
                            <i class="bi bi-database"></i> Vetor: ${vecSim}%
                        </span>` : ''}
                    </div>
                `;
            }

            return this.replaceTemplate(this.#productTemplate, {
                id: product.id,
                name: product.name,
                category: product.category,
                price: product.price,
                color: product.color,
                scoreBadge,
                product: JSON.stringify(product)
            });
        }).join('');

        this.#productList.innerHTML = html;
        this.attachBuyButtonListeners();

        // Disable all buttons by default
        this.setButtonsState(disableButtons);
    }

    setButtonsState(disabled) {
        if (!this.#buttons) {
            this.#buttons = document.querySelectorAll('.buy-now-btn');
        }
        this.#buttons.forEach(button => {
            button.disabled = disabled;
        });
    }

    attachBuyButtonListeners() {
        this.#buttons = document.querySelectorAll('.buy-now-btn');
        this.#buttons.forEach(button => {
            button.addEventListener('click', (event) => {
                const product = JSON.parse(button.dataset.product);
                const originalText = button.innerHTML;

                button.innerHTML = '<i class="bi bi-check-circle-fill"></i> Added';
                button.classList.remove('btn-primary');
                button.classList.add('btn-success');
                setTimeout(() => {
                    button.innerHTML = originalText;
                    button.classList.remove('btn-success');
                    button.classList.add('btn-primary');
                }, 500);
                this.#onBuyProduct(product, button);
            });
        });
    }
}
