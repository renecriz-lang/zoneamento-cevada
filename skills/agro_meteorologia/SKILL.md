# SKILL: Agrometeorologia com Decêndios (ZARC-Ready)

## Descrição
Processa dados meteorológicos de estações (manuais + automáticas) em **decêndios** (padrão ZARC - Zoneamento Agrícola de Risco Climático). Preenche dados faltantes usando IDW (Inverse Distance Weighting) e média climatológica, gera TIFs interpolados por grid e marca quais dados foram adicionados.

## Entrada (Input)
- **Pasta de estações manuais:** CSVs com metadados (Código, Nome, Lat, Lon, Alt) + dados diários
- **Pasta de estações automáticas:** CSVs com mesma estrutura
- **Shapefile de municípios** (opcional): Para futuro recorte

## Saída (Output)
- **Estações mapeadas:** XLSX com códigos + vizinhos mais próximos
- **Estações preenchidas:** CSVs por estação com dados tratados + coluna de origem (ORIGINAL/IDW/MEDIA_MENSAL)
- **TIFs interpolados:** Grid com dados agregados (108 decêndios × 3 variáveis)
- **PNGs (opcional):** Visualização colorida dos TIFs
- **JSONs e CSVs:** Resumos estatísticos

## Parâmetros Interativos

O script pergunta:

1. **Qual é o intervalo que quer usar?**
   - Opções: Semana, Decêndio, Pentada, Mês
   - Padrão: **Decêndio**

2. **Quantas estações vizinhas usar no IDW?**
   - Padrão: 3
   - Aceitável: 3-15

3. **Qual variável processar?**
   - Opções: Precipitação, Temperatura Média, Temperatura Max, Temperatura Min
   - Ou TODAS

4. **Gerar PNGs de visualização?**
   - Sim/Não (após TIFs prontos)

## Fluxo de Processamento

```
1. MAPEAMENTO
   └─ Carrega estações (manuais + automáticas)
   └─ Calcula distâncias Haversine
   └─ Identifica N vizinhos mais próximos
   └─ Salva XLSX com mapeamento

2. PREENCHIMENTO IDW
   └─ Para cada NULL encontrado:
   └─ Busca dado na mesma data nos N vizinhos
   └─ Aplica IDW: V = Σ(Vi/di²) / Σ(1/di²)
   └─ Marca com "IDW" em coluna de origem
   
3. PREENCHIMENTO MÉDIA CLIMATOLÓGICA
   └─ Para NULLs ainda faltando:
   └─ Calcula média do intervalo (decênio/semana/mês)
   └─ Preenche com valor climatológico
   └─ Marca com "MEDIA_[INTERVALO]"

4. VALIDAÇÃO DE QUALIDADE
   └─ Se decêndio tem >2 flags ruins → marca como "DESCARTADA"
   └─ Se >2 flags → não entra no IDW final

5. INTERPOLAÇÃO IDW POR GRID
   └─ Cria grid 0.02° (~2km) cobrindo Brasil
   └─ Para cada ponto: aplica IDW com estações usáveis
   └─ Gera TIF georreferenciado (EPSG:4326)
   └─ Total: calcular para o número de anos análisados e número de intervalo

6. VISUALIZAÇÃO (OPCIONAL: pergunte se quero fazer os desenhos pngs)
   └─ Converte TIFs em PNGs com colormap
   └─ Azul (seco/frio) → Verde → Vermelho (chuvoso/quente)
```

## Colunas Geradas nos CSVs Finalizados

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `Codigo_Estacao` | string | Código INMET da estação |
| `Nome_Estacao` | string | Nome da estação |
| `Latitude` | float | Coordenada Y (graus) |
| `Longitude` | float | Coordenada X (graus) |
| `Altitude` | float | Altitude em metros |
| `Tipo` | string | Manual / Automática |
| `Data` | date | Data da observação |
| `Valor_Original` | float | Dado bruto (pode ser null) |
| `Valor_Preenchido` | float | Dado final (sem null) |
| `Flag_Origem` | string | **ORIGINAL** / **IDW** / **MEDIA_MENSAL** / **NAO_PREENCHIVEL** |
| `Vizinho_Usado` | string | Se IDW: código da estação usada (ou "3 vizinhos") |
| `Distancia_Km` | float | Se IDW: distância até vizinho |

## Estrutura de Saída

```
Pasta de Trabalho/
├── Mapeamento_Estacoes.xlsx
│   ├── Todas Estações
│   ├── Estações + Vizinhos
│   └── Resumo Estatístico
│
├── Estacoes_Preenchidas/
│   ├── A001_preenchido.csv
│   ├── A002_preenchido.csv
│   └── ... (813 estações)
│
├── Estacoes_Finalizadas_Decendios/
│   ├── TIF_IDW_Decendios/
│   │   ├── Precip_Decendio_2023_01.tif
│   │   ├── Temp_Media_Decendio_2023_01.tif
│   │   ├── Temp_Max_Decendio_2023_01.tif
│   │   └── Temp_Min_Decendio_2023_01.tif
│   │   └── ... (108 decêndios)
│   │
│   ├── PNG_IDW_Decendios/ (se solicitado)
│   │   ├── Precip_Decendio_2023_01.png
│   │   └── ...
│   │
│   ├── Resumo_IDW_Decendios.json
│   └── Relatorio_Decendios.csv
│
└── Validacao_Decendios.csv
```

## Exemplo de Uso

```python
# 1. Carregar dados
estacoes = carregar_estacoes(
    pasta_manuais="estações_meteorológicas_BR_2023_2025/",
    pasta_automaticas="dados_estacoes_2023-2025_automáticas/"
)

# 2. Mapear vizinhos
vizinhos = mapear_vizinhos(estacoes, n_vizinhos=3)

# 3. Preencher NULLs
dados_preenchidos = preencher_idw_e_media(
    estacoes=estacoes,
    vizinhos=vizinhos,
    intervalo="decendio",  # ou "semana", "pentada", "mes"
    variavel="precipitacao"  # ou "temp_media", "temp_max", "temp_min"
)

# 4. Gerar TIFs
tifs = interpolar_idw_grid(
    dados_preenchidos,
    resolucao=0.02,
    raio_busca=100  # km
)

# 5. Gerar PNGs (opcional)
pngs = gerar_visualizacao(tifs, colormap="precip")
```

## Validação e Flags de Confiança

### Flag "ORIGINAL"
- Dado vinha do CSV original (100% confiável)

### Flag "IDW"
- Preenchido com dados de vizinhos próximos
- Confiança varia com distância:
  - `< 50 km` → Verde (alta confiança)
  - `50-100 km` → Amarelo (média)
  - `> 100 km` → Vermelho (baixa)

### Flag "MEDIA_MENSAL"
- Preenchido com climatologia do período
- Todos os vizinhos também faltavam

### Flag "NAO_PREENCHIVEL"
- Impossível preencher (mesmo a média climatológica falta)
- Raro, mas marca transparência

## Nota Importante

**Este skill gera TIFs prontos para serem recortados por municípios, regiões ou áreas de interesse** usando a próxima skill: `agro-municipios-decendios` (em desenvolvimento).

## Requisitos
- Python 3.8+
- rasterio, geopandas, scipy, numpy, pandas
- Shapefile de municípios (opcional, para validação)

## Referências
- **ZARC:** Zoneamento Agrícola de Risco Climático (MAPA)
- **Decêndios:** Padrão da Embrapa e INMET
- **IDW:** Técnica recomendada pela OMM (Organização Meteorológica Mundial)
