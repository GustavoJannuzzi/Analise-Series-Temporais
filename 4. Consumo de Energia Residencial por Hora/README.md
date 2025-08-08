# Consumo de Energia Residencial por Hora

**Tema:** Previsão de uso energético, automação residencial  
**Tipo de modelo sugerido:** SARIMA, Prophet, LSTM  
**Características:** Sazonalidade diária e semanal clara  

**Fonte:** [UCI - Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

## Descrição

Este dataset contém **2.075.259 medições** coletadas em uma residência localizada em Sceaux (7km de Paris, França) entre **dezembro de 2006 e novembro de 2010** (47 meses). Os dados representam o consumo elétrico detalhado da residência com medições realizadas a cada minuto.

## Estrutura dos Dados

O arquivo está no formato `.txt` com dados separados por ponto e vírgula (`;`). O dataset contém as seguintes variáveis:

| Coluna | Descrição | Unidade |
|--------|-----------|---------|
| `Date` | Data no formato dd/mm/yyyy | - |
| `Time` | Hora no formato hh:mm:ss | - |
| `Global_active_power` | Potência ativa global consumida pela residência | kilowatts |
| `Global_reactive_power` | Potência reativa global consumida pela residência | kilowatts |
| `Voltage` | Voltagem média por minuto | volts |
| `Global_intensity` | Intensidade de corrente global média por minuto | amperes |
| `Sub_metering_1` | Medição da cozinha (forno, micro-ondas, lava-louças, etc.) | watt-hora |
| `Sub_metering_2` | Medição da lavanderia (máquina de lavar, secadora, etc.) | watt-hora |
| `Sub_metering_3` | Medição do aquecedor elétrico e ar condicionado | watt-hora |

## Carregamento dos Dados

### Código para carregar o dataset no pandas:

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Carregar o dataset do arquivo .txt
df = pd.read_csv('household_power_consumption.txt', 
                 sep=';',           # Separador é ponto e vírgula
                 na_values='?',     # Valores ausentes são representados por '?'
                 low_memory=False)

print("Shape do dataset:", df.shape)
print("Primeiras linhas:")
print(df.head())
```

### Tratamento inicial dos dados:

```python
# Verificar tipos de dados
print("Tipos de dados:")
print(df.dtypes)

# Verificar valores ausentes
print("\nValores ausentes por coluna:")
print(df.isnull().sum())

# Converter colunas numéricas (elas são lidas como object devido aos valores '?')
numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print("Dados após conversão:")
print(df.dtypes)
```

## Preparação para Análise de Séries Temporais

### 1. Criação do Índice Temporal

```python
# Combinar Date e Time em uma única coluna datetime
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], 
                                format='%d/%m/%Y %H:%M:%S')

# Definir DateTime como índice
df.set_index('DateTime', inplace=True)

# Remover colunas originais de data e hora
df = df.drop(['Date', 'Time'], axis=1)

print("Período dos dados:")
print(f"Início: {df.index.min()}")
print(f"Fim: {df.index.max()}")
print(f"Frequência: {df.index.freq}")
```

### 2. Agregação por Hora 

```python
# Agregar dados por hora (média dos valores por hora)
df_hourly = df.resample('H').mean()

print("Dados agregados por hora:")
print(df_hourly.head())
print(f"Novo shape: {df_hourly.shape}")
```

### 3. Tratamento de Valores Ausentes

```python
# Verificar padrão de valores ausentes
missing_info = df_hourly.isnull().sum()
print("Valores ausentes após agregação:")
print(missing_info)

# Opções para tratamento:
# Opção 1: Interpolação linear
df_filled = df_hourly.interpolate(method='linear')

# Opção 2: Forward fill para pequenos gaps
# df_filled = df_hourly.fillna(method='ffill', limit=3)

# Opção 3: Remover linhas com valores ausentes
# df_filled = df_hourly.dropna()

print("Valores ausentes após tratamento:")
print(df_filled.isnull().sum())
```

## Análise Exploratória Inicial

### Visualização da Série Principal

```python
import matplotlib.pyplot as plt

# Gráfico da potência ativa global
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(df_hourly.index, df_hourly['Global_active_power'])
plt.title('Consumo de Energia - Potência Ativa Global')
plt.ylabel('Kilowatts')
plt.grid(True, alpha=0.3)

# Gráfico dos sub-medidores
plt.subplot(2, 1, 2)
plt.plot(df_hourly.index, df_hourly['Sub_metering_1'], label='Cozinha', alpha=0.7)
plt.plot(df_hourly.index, df_hourly['Sub_metering_2'], label='Lavanderia', alpha=0.7)
plt.plot(df_hourly.index, df_hourly['Sub_metering_3'], label='Aquecimento/AC', alpha=0.7)
plt.title('Consumo por Sub-medidores')
plt.ylabel('Watt-hora')
plt.xlabel('Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### Análise de Sazonalidade

```python
# Adicionar variáveis temporais para análise
df_analysis = df_hourly.copy()
df_analysis['Hour'] = df_analysis.index.hour
df_analysis['DayOfWeek'] = df_analysis.index.dayofweek
df_analysis['Month'] = df_analysis.index.month
df_analysis['Year'] = df_analysis.index.year

# Padrão por hora do dia
hourly_pattern = df_analysis.groupby('Hour')['Global_active_power'].mean()
plt.figure(figsize=(12, 4))
plt.plot(hourly_pattern.index, hourly_pattern.values)
plt.title('Padrão de Consumo por Hora do Dia')
plt.xlabel('Hora')
plt.ylabel('Potência Média (kW)')
plt.grid(True, alpha=0.3)
plt.show()
```

## Observações Importantes

- **Valores ausentes:** Aproximadamente 1,25% dos registros possuem valores ausentes
- **Frequência original:** Medições a cada minuto (bem granular)
- **Recomendação:** Agregar por hora ou dia para reduzir ruído e facilitar análise
- **Período:** 47 meses de dados (dezembro/2006 a novembro/2010)
- **Sazonalidades:** Padrões diários, semanais e sazonais claros
- **Equipamentos não medidos:** A diferença `(global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3)` representa outros equipamentos elétricos

## Sugestões para Análise

1. **Análise de uma variável:** Focar na `Global_active_power`
2. **Análise multivariada:** Incluir sub-medidores e variáveis climáticas
3. **Diferentes granularidades:** Análise horária, diária ou mensal
4. **Detecção de anomalias:** Identificar padrões de consumo anômalos
5. **Previsão:** Implementar modelos SARIMA, Prophet ou LSTM para diferentes horizontes de previsão