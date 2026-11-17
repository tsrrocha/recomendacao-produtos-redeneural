// Configuração central da API do Banco de Dados Vetorial
// Se a página estiver rodando no browser-sync (porta 3000), conecta na porta da API (3002 ou 3001)
// Caso esteja acessando diretamente o servidor Express, usa a mesma origem (/api).
function getApiBaseUrl() {
    if (typeof window !== 'undefined' && window.location.port === '3000') {
        return 'http://localhost:3002/api';
    }
    return typeof window !== 'undefined' 
        ? `${window.location.origin}/api` 
        : 'http://localhost:3002/api';
}

export const API_BASE_URL = getApiBaseUrl();
console.log(`[Config] API do Banco de Dados Vetorial configurada para: ${API_BASE_URL}`);
