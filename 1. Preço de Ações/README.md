# Preço de Ações de Big Tech (Apple, Amazon, NVIDIA...)

**Tema:** Previsão financeira e investimento  
**Tipo de modelo sugerido:** ARIMA, LSTM, GARCH  
**Características da série:** Volátil, tendência + possível sazonalidade

**Fonte:** Yahoo Finance via biblioteca `yfinance`

## Descrição

Este dataset contém dados históricos de preços de ações.
Fica a critério da equipe escolher qual ação será analisada. Dê preferência para ações de tecnologia para aproveitar a volatilidade da série.(Ex: AAPL, NVDA, AMZN, MSFT, GOOGL...)

As séries temporais de preços de ações são caracterizadas por alta volatilidade, presença de tendências de longo prazo e possível sazonalidade relacionada a eventos corporativos (lançamento de produtos, divulgação de resultados trimestrais, etc.).

## Instalação e Configuração

### Instalar a biblioteca yfinance:

```bash
pip install yfinance
```

## Download dos Dados

### Código básico para download:

```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Escolher uma das opções abaixo (ou outra big tech):
# Apple
df = yf.download('AAPL', start='2015-01-01', end='2025-01-01')

print("Shape dos dados:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())
```

## Estrutura dos Dados

O dataset retornado pelo `yfinance` contém as seguintes colunas:

| Coluna | Descrição | Unidade |
|--------|-----------|---------|
| `Open` | Preço de abertura do dia | USD ($) |
| `High` | Preço mais alto do dia | USD ($) |
| `Low` | Preço mais baixo do dia | USD ($) |
| `Close` | Preço de fechamento do dia | USD ($) |
| `Adj Close` | Preço de fechamento ajustado (dividendos e splits) | USD ($) |
| `Volume` | Volume de ações negociadas no dia | Quantidade |

### Explicação das colunas:

- **Open:** Primeiro preço negociado quando o mercado abre
- **High:** Maior preço atingido durante o pregão
- **Low:** Menor preço atingido durante o pregão  
- **Close:** Último preço negociado quando o mercado fecha
- **Adj Close:** Preço de fechamento ajustado para dividendos e desdobramentos (splits)
- **Volume:** Total de ações que mudaram de mãos durante o dia

## Análise Exploratória Inicial

### Verificação dos dados:

```python
# Informações básicas
print("Informações do dataset:")
print(df.info())

print("\nEstatísticas descritivas:")
print(df.describe())

print("\nVerificar valores ausentes:")
print(df.isnull().sum())

print("\nPeríodo dos dados:")
print(f"Data inicial: {df.index.min()}")
print(f"Data final: {df.index.max()}")
print(f"Total de dias úteis: {len(df)}")
```

### Visualização da série temporal:

```python
# Gráfico dos preços
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Preço de fechamento
axes[0,0].plot(df.index, df['Close'])
axes[0,0].set_title('Preço de Fechamento')
axes[0,0].set_ylabel('Preço ($)')
axes[0,0].grid(True, alpha=0.3)

# Preço ajustado
axes[0,1].plot(df.index, df['Adj Close'])
axes[0,1].set_title('Preço Ajustado')
axes[0,1].set_ylabel('Preço ($)')
axes[0,1].grid(True, alpha=0.3)

# Volume
axes[1,0].plot(df.index, df['Volume'])
axes[1,0].set_title('Volume de Negociação')
axes[1,0].set_ylabel('Volume')
axes[1,0].grid(True, alpha=0.3)

# Volatilidade diária (High - Low)
df['Daily_Range'] = df['High'] - df['Low']
axes[1,1].plot(df.index, df['Daily_Range'])
axes[1,1].set_title('Volatilidade Diária (High - Low)')
axes[1,1].set_ylabel('Diferença de Preço ($)')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Preparação para Análise de Séries Temporais

### 1. Escolha da Variável Principal

```python
# Para análise de séries temporais, geralmente usa-se:
# Opção 1: Preço ajustado (recomendado)
price_series = df['Adj Close'].copy()

# Opção 2: Log dos preços (para estabilizar variância)
log_price_series = np.log(df['Adj Close']).copy()

# Opção 3: Retornos (já estacionária)
return_series = df['Daily_Return'].dropna().copy()

print("Série escolhida para análise:")
print(f"Preços ajustados - Shape: {price_series.shape}")
print(price_series.head())
```

### 2. Análise de Estacionariedade

```python
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries, title):
    print(f'Teste de Estacionariedade para {title}:')
    
    # Teste ADF
    result = adfuller(timeseries.dropna())
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.6f}')
    
    if result[1] <= 0.05:
        print("Resultado: Série é ESTACIONÁRIA")
    else:
        print("Resultado: Série NÃO é estacionária")
    print("-" * 50)

# Testar diferentes transformações
test_stationarity(price_series, 'Preços Originais')
test_stationarity(log_price_series, 'Log dos Preços')
test_stationarity(return_series, 'Retornos Diários')

# Diferenciação dos preços se necessário
price_diff = price_series.diff().dropna()
test_stationarity(price_diff, 'Primeira Diferença dos Preços')
```

### 3. Análise de Sazonalidade e Tendência

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposição da série temporal (usando preços)
decomposition = seasonal_decompose(price_series.dropna(), model='multiplicative', period=252)

# Plotar decomposição
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

decomposition.observed.plot(ax=axes[0], title='Série Original')
decomposition.trend.plot(ax=axes[1], title='Tendência')
decomposition.seasonal.plot(ax=axes[2], title='Sazonalidade')
decomposition.resid.plot(ax=axes[3], title='Resíduos')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4. Análise de Volatilidade

```python
# Volatilidade móvel (janela de 30 dias)
df['Volatility_30d'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252)

# Volatilidade realizada
plt.figure(figsize=(15, 6))
plt.plot(df.index, df['Volatility_30d'] * 100)
plt.title('Volatilidade Móvel (30 dias) - Anualizada')
plt.ylabel('Volatilidade (%)')
plt.xlabel('Data')
plt.grid(True, alpha=0.3)
plt.show()

print("Estatísticas de volatilidade:")
print(f"Volatilidade média: {df['Volatility_30d'].mean()*100:.2f}%")
print(f"Volatilidade máxima: {df['Volatility_30d'].max()*100:.2f}%")
print(f"Volatilidade mínima: {df['Volatility_30d'].min()*100:.2f}%")
```

## Observações Importantes

- **Escolha da ação:** Cada ação tem características diferentes (AAPL mais estável, NVDA mais volátil)
- **Preço vs Retorno:** Preços não são estacionários, retornos geralmente são
- **Adj Close:** Sempre use preço ajustado para análise temporal (considera splits e dividendos)
- **Fins de semana:** Dados só incluem dias úteis de negociação
- **Eventos corporativos (Ruído):** Lançamentos de produtos, resultados trimestrais afetam os preços
- **Volatilidade:** Característica fundamental dos mercados financeiros

## Sugestões para Análise

1. **ARIMA/SARIMA:** Para modelagem de preços ou retornos
2. **GARCH:** Para modelagem de volatilidade condicional
3. **LSTM/GRU:** Para capturar padrões não-lineares complexos