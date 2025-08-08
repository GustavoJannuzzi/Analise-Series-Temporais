# 3. Frequência de Logs de Erro/Alerta de Sistema Web

## Descrição Geral
Este dataset contém dados sintéticos de monitoramento de logs de erro e alerta de um sistema web durante um período de 365 dias (1 ano), com medições a cada hora, totalizando 8.760 observações. Os dados simulam um ambiente de produção real com características típicas de sistemas web em operação.

## Características do Dataset

### Colunas:
1. **timestamp** (datetime): Data e hora da observação no formato YYYY-MM-DD HH:MM:SS
2. **error_count** (integer): Número total de logs de erro registrados na hora
3. **alert_count** (integer): Número total de alertas gerados na hora
4. **system_load** (float): Percentual de carga do sistema (0-100%)
5. **response_time** (float): Tempo médio de resposta em milissegundos
6. **active_users** (integer): Número de usuários ativos no sistema
7. **day_of_week** (integer): Dia da semana (0=Segunda, 6=Domingo)
8. **hour** (integer): Hora do dia (0-23)
9. **is_weekend** (boolean): Indica se é fim de semana (True/False)
10. **maintenance_window** (boolean): Indica se está em janela de manutenção (True/False)

### Adequação para Modelos:

#### ARMA:
- Presença de autocorrelação temporal
- Componentes autoregressivos e de média móvel
- Estacionaridade após diferenciação sazonal

#### LSTM:
- Padrões temporais complexos e não-lineares
- Múltiplas sazonalidades sobrepostas
- Dependências de longo prazo
- Variáveis exógenas (carga, usuários, etc.)

