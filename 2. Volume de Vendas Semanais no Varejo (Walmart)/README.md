# 2. Volume de Vendas Semanais no Varejo (Walmart)

**Tema:** Previsão de demanda e estoque  
**Tipo de modelo sugerido:** SARIMA, Prophet, LSTM  
**Características:** Forte sazonalidade (Black Friday, Natal, férias)  

**Fonte:** [Kaggle - Walmart Sales Dataset](https://www.kaggle.com/datasets/mikhail1681/walmart-sales)

## Descrição

Este dataset contém dados de vendas semanais de uma das maiores redes de varejo do mundo (Walmart).

## Estrutura dos Dados

O dataset contém as seguintes colunas:

| Coluna | Descrição | Unidade/Tipo |
|--------|-----------|--------------|
| `Store` | Número identificador da loja | Inteiro |
| `Date` | Data de início da semana de vendas | Data (YYYY-MM-DD) |
| `Weekly_Sales` | Volume de vendas semanais | Dólares americanos ($) |
| `Holiday_Flag` | Indicador de presença de feriado (1 = feriado, 0 = sem feriado) | Binário (0/1) |
| `Temperature` | Temperatura do ar na região da loja | Fahrenheit (°F) |
| `Fuel_Price` | Custo do combustível na região | Dólares por galão ($/gal) |
| `CPI` | Índice de Preços ao Consumidor | Índice |
| `Unemployment` | Taxa de desemprego na região | Percentual (%) |

## Carregamento dos Dados

### Código para carregar o dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv('walmart_sales.csv')

print("Shape do dataset:", df.shape)
print("\nPrimeiras linhas:")
print(df.head())

print("\nInformações gerais:")
print(df.info())
```

### Verificação inicial dos dados:

```python
# Verificar valores ausentes
print("Valores ausentes por coluna:")
print(df.isnull().sum())

# Estatísticas descritivas
print("\nEstatísticas descritivas:")
print(df.describe())

# Verificar período dos dados
print(f"\nPeríodo dos dados:")
print(f"Data inicial: {df['Date'].min()}")
print(f"Data final: {df['Date'].max()}")

# Verificar número de lojas
print(f"\nNúmero de lojas: {df['Store'].nunique()}")
print(f"Lojas: {sorted(df['Store'].unique())}")
```

## Preparação para Análise de Séries Temporais

### 1. Tratamento do Índice Temporal

```python
# Converter Date para datetime
df['Date'] = pd.to_datetime(df['Date'])

# Verificar se há lojas suficientes para análise individual ou agregada
print("Número de observações por loja:")
print(df['Store'].value_counts().sort_index())

# Opção A: Análise agregada (todas as lojas)
df_total = df.groupby('Date').agg({
    'Weekly_Sales': 'sum',
    'Temperature': 'mean',
    'Fuel_Price': 'mean', 
    'CPI': 'mean',
    'Unemployment': 'mean',
    'Holiday_Flag': 'max'  # Se qualquer loja tem feriado, considera feriado
}).reset_index()

df_total.set_index('Date', inplace=True)
print("\nDados agregados por data:")
print(df_total.head())
```

### 2. Análise por Loja Individual (Opcional)

```python
# Opção B: Análise de uma loja específica
store_id = 1  # Escolher uma loja para análise
df_store = df[df['Store'] == store_id].copy()
df_store.set_index('Date', inplace=True)
df_store = df_store.sort_index()

print(f"Dados da loja {store_id}:")
print(df_store.head())
print(f"Período: {df_store.index.min()} até {df_store.index.max()}")
```

### 3. Identificação de Feriados e Sazonalidade

```python
# Analisar padrão de feriados
holidays = df_total[df_total['Holiday_Flag'] == 1].index
print("Datas com feriados:")
print(holidays.tolist())

# Adicionar variáveis temporais
df_analysis = df_total.copy()
df_analysis['Year'] = df_analysis.index.year
df_analysis['Month'] = df_analysis.index.month
df_analysis['Week'] = df_analysis.index.isocalendar().week
df_analysis['Quarter'] = df_analysis.index.quarter

print("Variáveis temporais adicionadas:")
print(df_analysis[['Year', 'Month', 'Week', 'Quarter']].head())
```

## Análise Exploratória

### Visualização da Série Temporal Principal

```python
# Gráfico das vendas semanais
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.plot(df_total.index, df_total['Weekly_Sales'], linewidth=1)
plt.title('Volume de Vendas Semanais - Walmart (Todas as Lojas)')
plt.ylabel('Vendas ($)')
plt.grid(True, alpha=0.3)

# Destacar feriados
for holiday in holidays:
    plt.axvline(x=holiday, color='red', linestyle='--', alpha=0.6)

plt.subplot(2, 1, 2)
plt.plot(df_total.index, df_total['Weekly_Sales'], linewidth=1)
plt.title('Zoom - Vendas com Tendência')
plt.ylabel('Vendas ($)')
plt.xlabel('Data')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Análise de Sazonalidade

```python
# Padrão mensal
monthly_sales = df_analysis.groupby('Month')['Weekly_Sales'].mean()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
monthly_sales.plot(kind='bar')
plt.title('Vendas Médias por Mês')
plt.ylabel('Vendas Médias ($)')
plt.xticks(rotation=45)

# Padrão por trimestre
plt.subplot(1, 3, 2)
quarterly_sales = df_analysis.groupby('Quarter')['Weekly_Sales'].mean()
quarterly_sales.plot(kind='bar', color='orange')
plt.title('Vendas Médias por Trimestre')
plt.ylabel('Vendas Médias ($)')

# Comparação feriados vs dias normais
plt.subplot(1, 3, 3)
holiday_comparison = df_total.groupby('Holiday_Flag')['Weekly_Sales'].mean()
holiday_comparison.plot(kind='bar', color=['blue', 'red'])
plt.title('Vendas: Feriado vs Normal')
plt.ylabel('Vendas Médias ($)')
plt.xticks([0, 1], ['Normal', 'Feriado'], rotation=0)

plt.tight_layout()
plt.show()
```

### Análise de Correlação com Fatores Externos

```python
# Matriz de correlação
correlation_vars = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']
correlation_matrix = df_total[correlation_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, fmt='.3f')
plt.title('Matriz de Correlação - Fatores que Influenciam as Vendas')
plt.show()

print("Correlações com Weekly_Sales:")
correlations = correlation_matrix['Weekly_Sales'].sort_values(ascending=False)
print(correlations)
```

### Análise de Outliers e Eventos Especiais

```python
# Identificar semanas com vendas muito altas ou baixas
Q1 = df_total['Weekly_Sales'].quantile(0.25)
Q3 = df_total['Weekly_Sales'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_total[(df_total['Weekly_Sales'] < lower_bound) | 
                   (df_total['Weekly_Sales'] > upper_bound)]

print("Semanas com vendas atípicas:")
print(outliers[['Weekly_Sales', 'Holiday_Flag', 'Temperature']].head(10))

# Visualizar outliers
plt.figure(figsize=(12, 6))
plt.boxplot(df_total['Weekly_Sales'])
plt.title('Distribuição das Vendas Semanais - Identificação de Outliers')
plt.ylabel('Vendas ($)')
plt.show()
```

## Observações Importantes

- **Múltiplas lojas:** Dataset contém várias lojas - você pode decidir entre análise agregada ou individual
- **Sazonalidade forte:** Padrões claros em feriados (Black Friday, Natal) e trimestres
- **Fatores externos:** Temperatura, combustível, CPI e desemprego podem influenciar vendas
- **Outliers:** Identificar e tratar semanas com vendas atípicas (promoções, eventos especiais)
- **Frequência:** Dados semanais são ideais para capturar sazonalidade mensal e trimestral
- **Feriados:** Variable importante para capturar picos de vendas

## Sugestões para Análise

1. **Análise agregada:** Somar vendas de todas as lojas por semana
2. **Análise por loja:** Focar em uma loja específica com mais dados
3. **Modelos univariados:** SARIMA com sazonalidade semanal/mensal
4. **Detecção de eventos:** Análise do impacto de feriados e promoções especiais
