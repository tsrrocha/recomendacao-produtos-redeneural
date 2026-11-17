import { UserController } from './controller/UserController.js';
import { ProductController } from './controller/ProductController.js';
import { ModelController } from './controller/ModelTrainingController.js';
import { TFVisorController } from './controller/TFVisorController.js';
import { TFVisorView } from './view/TFVisorView.js';
import { UserService } from './service/UserService.js';
import { ProductService } from './service/ProductService.js';
import { UserView } from './view/UserView.js';
import { ProductView } from './view/ProductView.js';
import { ModelView } from './view/ModelTrainingView.js';
import Events from './events/events.js';
import { WorkerController } from './controller/WorkerController.js';
import { API_BASE_URL } from './config.js';

// Status do Banco de Dados Vetorial na Interface
async function updateDbStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        if (response.ok) {
            const data = await response.json();
            const vectorsText = `${data.productsWithVectors}/${data.totalProducts} produtos com vetores salvos | ${data.totalUsers} usuários | ${data.totalPurchases} compras no BD`;
            const statusElem = document.getElementById('dbVectorsCount');
            const engineElem = document.getElementById('dbEngineText');
            if (statusElem) statusElem.textContent = vectorsText;
            if (engineElem) engineElem.textContent = 'SQLite Vetorial (node:sqlite)';
            return;
        }
    } catch (e) {
        console.warn('Não foi possível conectar ao servidor de API:', e.message);
    }
    const statusElem = document.getElementById('dbVectorsCount');
    if (statusElem) statusElem.textContent = 'Modo local offline (inicie "npm start")';
}

// Configura botão de reset do banco de dados
const resetBtn = document.getElementById('dbResetBtn');
if (resetBtn) {
    resetBtn.addEventListener('click', async () => {
        if (!confirm('Deseja realmente resetar o banco de dados para os valores originais do JSON?')) return;
        try {
            const res = await fetch(`${API_BASE_URL}/database/reset`, { method: 'POST' });
            if (res.ok) {
                alert('Banco de dados restaurado com sucesso! A página será recarregada.');
                window.location.reload();
            }
        } catch (e) {
            alert('Erro ao resetar banco: ' + e.message);
        }
    });
}

// Inicializa status
updateDbStatus();

// Create shared services
const userService = new UserService();
const productService = new ProductService();

// Create views
const userView = new UserView();
const productView = new ProductView();
const modelView = new ModelView();
const tfVisorView = new TFVisorView();
const mlWorker = new Worker('/src/workers/modelTrainingWorker.js', { type: 'module' });

// Set up worker message handler
const w = WorkerController.init({
    worker: mlWorker,
    events: Events
});

// Atualiza o status do banco quando o treino terminar e os embeddings forem persistidos
Events.onTrainingComplete(() => {
    setTimeout(updateDbStatus, 500);
});

const users = await userService.getDefaultUsers();
w.triggerTrain(users);

ModelController.init({
    modelView,
    userService,
    events: Events,
});

TFVisorController.init({
    tfVisorView,
    events: Events,
});

ProductController.init({
    productView,
    userService,
    productService,
    events: Events,
});

const userController = UserController.init({
    userView,
    userService,
    productService,
    events: Events,
});

userController.renderUsers({
    "id": 99,
    "name": "Josézin da Silva",
    "age": 30,
    "purchases": []
});
