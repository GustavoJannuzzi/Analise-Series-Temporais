# 5. Taxa de Crimes em Chicago
⦁	Tema: Segurança pública, predição de incidentes
⦁	Tipo de modelo sugerido: ARIMA, GARCH
⦁	Características: Tendências, eventos pontuais (feriados, fins de semana)
Fonte: [Chicago Crime Dataset](https://data.cityofchicago.org/stories/s/Crimes-2001-to-present-Dashboard/5cd6-ry5g)

## Descrição

Este dataset contém informações sobre incidentes de crimes reportados na cidade de Chicago. Ele foi extraído do sistema Chicago Data Portal. A base de dados inclui uma variedade de incidentes criminosos, onde os dados são específicos para cada vítima.

## Estrutura de Dados

O dataset contém as seguintes colunas:

| Coluna | Descrição |
|--------|-----------|
| `ID` | Identificador único para o registro |
| `Case Number` | Número de registro do incidente, único para cada caso |
| `Date` | Data do incidente (às vezes uma estimativa) |
| `Block` | Endereço parcialmente redigido onde o incidente ocorreu (no nível do bloco) |
| `IUCR` | Código do Uniform Crime Reporting de Illinois, vinculado ao tipo principal e descrição do crime |
| `Primary Type` | Descrição principal do código IUCR |
| `Description` | Descrição secundária do código IUCR |
| `Location Description` | Descrição do local do incidente |
| `Arrest` | Indica se houve prisão (True/False) |
| `Domestic` | Indica se o incidente foi relacionado à violência doméstica (True/False) |
| `Beat` | A área geográfica mais local onde o incidente ocorreu |
| `District` | O distrito policial onde o incidente ocorreu |
| `Ward` | O distrito do conselho da cidade onde o incidente ocorreu |
| `Community Area` | A área comunitária onde o incidente ocorreu |
| `FBI Code` | Classificação do crime conforme o sistema nacional de relatórios de incidentes da FBI |
| `X Coordinate` | Coordenada X do local do incidente (sistema de coordenadas de Illinois, com desvio para proteger privacidade) |
| `Y Coordinate` | Coordenada Y do local do incidente (sistema de coordenadas de Illinois, com desvio para proteger privacidade) |
| `Year` | Ano extraído da data do incidente |
| `Updated On` | Data da última atualização do registro |
| `Latitude` | Coordenada de latitude do local do incidente |
| `Longitude` | Coordenada de longitude do local do incidente |
| `Location` | Combinação de latitude e longitude para criação de mapas e análise geográfica |

---
## Guia de Tratamento dos Dados para Análise de Séries Temporais

### 1. Preparação e Indexação Temporal
Assumindo que sua base de dados esteja em pandas Data Frame, df.

```python
import pandas as pd
import numpy as np

# Converter a coluna Date para datetime
df['Date'] = pd.to_datetime(df['Date'])

# Criar índice temporal agrupado por dia
df_daily = df.set_index('Date').resample('D').size().reset_index()
df_daily.columns = ['Date', 'Crime_Count']

# Ordenar por data (caso necessário)
df_daily = df_daily.sort_values('Date')

# Definir Date como índice final
df_daily.set_index('Date', inplace=True)

print("Dados agrupados por dia:")
print(df_daily.head())
```

### 2. Limpeza dos Dados

Para análise de séries temporais, as seguintes colunas podem ser removidas do dataset original:

```python
# Colunas irrelevantes para análise temporal
colunas_remover = [
    'ID', 'Case Number', 'Block', 'IUCR', 'Location Description', 
    'Beat', 'FBI Code', 'X Coordinate', 'Y Coordinate', 'Updated On', 
    'Latitude', 'Longitude', 'Location'
]

# Manter apenas colunas relevantes para análise temporal
df_clean = df.drop(columns=colunas_remover)

print("Colunas mantidas para análise:")
print(df_clean.columns.tolist())
```

### 3. Opções de Análise

#### Opção A: Análise Geral (Todos os Crimes)
```python
# Contagem diária total de crimes
crimes_diarios = df_clean.groupby(df_clean['Date'].dt.date).size()
crimes_diarios.index = pd.to_datetime(crimes_diarios.index)

print(f"Período analisado: {crimes_diarios.index.min()} até {crimes_diarios.index.max()}")
print(f"Total de dias: {len(crimes_diarios)}")
```

#### Opção B: Análise por Tipo de Crime
```python
# Análise específica por tipo de crime
tipos_crime = df_clean['Primary Type'].value_counts()
print("Tipos de crime mais frequentes:")
print(tipos_crime.head(10))

# Exemplo: Análise de roubos (THEFT)
df_theft = df_clean[df_clean['Primary Type'] == 'THEFT']
theft_diario = df_theft.groupby(df_theft['Date'].dt.date).size()
theft_diario.index = pd.to_datetime(theft_diario.index)
```

#### Opção C: Análise por Localização
```python
# Análise por distrito policial
crimes_por_distrito = df_clean.groupby(['District', df_clean['Date'].dt.date]).size()

# Exemplo: Distrito específico
distrito_1 = df_clean[df_clean['District'] == 1]
distrito_1_diario = distrito_1.groupby(distrito_1['Date'].dt.date).size()
```

### 4. Verificação da Qualidade dos Dados

```python
# Verificar valores ausentes
print("Valores ausentes por coluna:")
print(df_clean.isnull().sum())

# Verificar dados duplicados
print(f"Registros duplicados: {df_clean.duplicated().sum()}")

# Estatísticas básicas da série temporal
print("\nEstatísticas da série temporal diária:")
print(crimes_diarios.describe())
```

### 5. Visualização Inicial

```python
import matplotlib.pyplot as plt

# Gráfico da série temporal
plt.figure(figsize=(15, 6))
plt.plot(crimes_diarios.index, crimes_diarios.values)
plt.title('Crimes Diários em Chicago')
plt.xlabel('Data')
plt.ylabel('Número de Crimes')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
```

### Sugestões para Análise

**Escolha uma das abordagens:**
1. **Análise geral:** Todos os tipos de crime agregados
2. **Análise específica:** Focar em um tipo de crime (ex: THEFT, ASSAULT, BATTERY)
3. **Análise geográfica:** Focar em distritos ou áreas específicas
4. **Análise comparativa:** Comparar diferentes tipos de crime ou regiões

**Modelos sugeridos:** ARIMA, GARCH, Prophet, ou modelos de Machine Learning para previsão de séries temporais.