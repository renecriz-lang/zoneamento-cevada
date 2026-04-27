# Exemplos de Uso - Skill Agrometeorologia

## Exemplo 1: Processamento Interativo Básico

```python
from agro_meteo import AgroMeteorologia

# Inicializa
agro = AgroMeteorologia(pasta_trabalho="./Meu_Projeto_Agrario")

# Interage com usuário
agro.perguntar_parametros()
agro.perguntar_gerar_pngs()

# Saída:
# - Parâmetros definidos (intervalo, vizinhos, variáveis, PNGs)
# - Pronto para processar
```

### Output Esperado:
```
══════════════════════════════════════════════════════════════════════════════
AGROMETEOROLOGIA COM DECÊNDIOS - CONFIGURAÇÃO
══════════════════════════════════════════════════════════════════════════════

1. Qual é o intervalo que quer usar?
   a) Decêndio (10 dias) - RECOMENDADO PARA ZARC
   b) Semana (7 dias)
   c) Pentada (5 dias)
   d) Mês completo
Escolha (a/b/c/d) [padrão: a]: a

2. Quantas estações vizinhas usar no IDW?
   (padrão: 3, aceita 3-15)
Escolha um número [padrão: 3]: 5

3. Qual variável processar?
   a) Precipitação
   b) Temperatura Média
   c) Temperatura Máxima
   d) Temperatura Mínima
   e) TODAS
Escolha (a/b/c/d/e) [padrão: a]: e

✓ Parâmetros definidos:
  - Intervalo: decendio
  - Vizinhos no IDW: 5
  - Variáveis: precipitacao, temperatura_media, temperatura_max, temperatura_min

══════════════════════════════════════════════════════════════════════════════
VISUALIZAÇÃO
══════════════════════════════════════════════════════════════════════════════

Gerar PNGs de visualização dos TIFs?
(Mais lento, mas útil para validação visual)

Gerar PNGs? (s/n) [padrão: s]: s
  ✓ PNGs serão gerados
```

---

## Exemplo 2: Processo Completo (Futuro)

```python
from agro_meteo import AgroMeteorologia

# 1. Inicializa
agro = AgroMeteorologia(pasta_trabalho="./Projeto_Cevada_Agraria")

# 2. Configuração interativa
agro.perguntar_parametros()  # Pergunta ao usuário
# Saída: intervalo="decendio", n_vizinhos=3, variaveis=["precipitacao"]

# 3. Carrega estações
estacoes = agro.carregar_estacoes(
    pasta_manuais="estações_meteorológicas_BR_2023_2025/",
    pasta_automaticas="dados_estacoes_2023-2025_automáticas/"
)
# Carrega 813 estações (196 manuais + 617 automáticas)

# 4. Mapeia vizinhos
vizinhos = agro.mapear_vizinhos(estacoes, n_vizinhos=3)
# Calcula distâncias Haversine, identifica 3 mais próximos

# 5. Preenche com IDW
estacoes = agro.preencher_com_idw(estacoes, coluna_valor="Precipitacao_Diaria_mm")
# Marca: Flag_Origem = "IDW" para dados preenchidos
# Mantém: Flag_Origem = "ORIGINAL" para dados já presentes

# 6. Preenche com média climatológica
estacoes = agro.preencher_com_media_climatologica(estacoes, coluna_valor="Precipitacao_Diaria_mm")
# Marca: Flag_Origem = "MEDIA_MENSAL" para dados ainda faltando

# 7. Valida decêndios
estacoes = agro.validar_decendios(estacoes, coluna_valor="Precipitacao_Diaria_mm", max_flags_ruins=2)
# Marca: Flag_Status_Decendio = "DESCARTADA_QUALIDADE" se >2 flags ruins

# 8. Gera relatório
agro.gerar_relatorio(estacoes)
# Output:
#   Estação A001 (Manual)
#     ORIGINAL:         8760
#     IDW:               145
#     MEDIA_MENSAL:       87
#     NAO_PREENCHIVEL:     4
#     Preenchimento:   99.95%

# 9. Gera TIFs interpolados
tifs = agro.interpolar_idw_grid(
    estacoes_usaveis_apenas=True,  # Ignora DESCARTADA_QUALIDADE
    resolucao_graus=0.02,
    raio_busca_km=100
)
# Saída: 108 TIFs (36 decênios × 3 anos)
# Caminho: ./Projeto_Cevada_Agraria/TIF_IDW_Decendios/

# 10. Gera PNGs (opcional)
if agro.gerar_pngs:
    pngs = agro.gerar_visualizacao(
        tifs=tifs,
        colormap="viridis",  # ou "precipitacao", "temperatura"
        dpi=150
    )
    # Saída: 108 PNGs com colormap
    # Caminho: ./Projeto_Cevada_Agraria/PNG_IDW_Decendios/

# 11. Gera JSONs de resumo
agro.gerar_json_resumo(tifs, output_arquivo="Resumo_IDW_Decendios.json")
# Estrutura: por decêndio, estações usadas, cobertura, estatísticas
```

---

## Exemplo 3: Processamento Customizado (Avançado)

```python
from agro_meteo import AgroMeteorologia

agro = AgroMeteorologia(pasta_trabalho="./Meu_Estudo_Regional")

# Define parâmetros diretamente (sem interação)
agro.intervalo = "semana"  # Em vez de decêndio
agro.n_vizinhos = 7  # Mais vizinhos para área com baixa densidade de estações
agro.variaveis = ["temperatura_max"]  # Só temperatura máxima
agro.gerar_pngs = True

# Processa
estacoes = agro.carregar_estacoes(
    pasta_manuais="./Dados_Regionais/estacoes_manuais/",
    pasta_automaticas="./Dados_Regionais/estacoes_automaticas/"
)

# ... resto do fluxo ...
```

---

## Exemplo 4: Integração com Próxima Skill (Municípios)

```python
from agro_meteo import AgroMeteorologia
from agro_municipios import AgroMunicipios  # (Futura skill)

# ETAPA 1: Processa dados meteorológicos (ESTE SKILL)
agro_meteo = AgroMeteorologia(pasta_trabalho="./Processamento")
# ... (configuração e processamento completo) ...
tifs_precipitacao = agro_meteo.interpolar_idw_grid(variavel="precipitacao")

# ETAPA 2: Recorta pelos municípios (PRÓXIMA SKILL)
agro_municipios = AgroMunicipios(pasta_trabalho="./Analise_Municipal")
municipios_json = agro_municipios.recortar_tifs_por_municipios(
    tifs=tifs_precipitacao,
    shapefile="BR_Municipios_2025/BR_Municipios_2025.shp"
)

# Output: JSON com precipitação por município × decêndio
# Estrutura compatível com análise de aptidão para cultivos
```

---

## Fluxo Completo Recomendado

```
┌─────────────────────────────────────────────────────────┐
│ 1. SKILL: agro-meteorologia-decendios (ESTE)           │
├─────────────────────────────────────────────────────────┤
│ Input:  CSVs de estações (manuais + automáticas)       │
│ Output: TIFs + PNGs + JSONs de decêndios               │
│                                                         │
│ - Carrega estações                                      │
│ - Mapeia vizinhos                                       │
│ - Preenche NULLs com IDW (marcando origem)             │
│ - Preenche restante com média climatológica             │
│ - Valida qualidade por decêndio                         │
│ - Gera TIFs interpolados (108 por variável)            │
│ - Gera PNGs (opcional)                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. SKILL: agro-municipios-decendios (PRÓXIMA)          │
├─────────────────────────────────────────────────────────┤
│ Input:  TIFs + Shapefile de municípios                  │
│ Output: JSON com dados agregados por município          │
│                                                         │
│ - Recorta TIFs pelos polígonos municipais              │
│ - Calcula estatísticas (média, min, max)                │
│ - Gera JSON com 5.573 municípios × 108 decêndios       │
└─────────────────────────────────────────────────────────┘
```

---

## Estrutura de Pastas Gerada

```
./Projeto_Cevada_Agraria/
├── Mapeamento_Estacoes.xlsx
│   ├── Todas Estações
│   ├── Estações + Vizinhos (3 mais próximos)
│   └── Resumo Estatístico
│
├── Estacoes_Preenchidas/
│   ├── A001_preenchido.csv
│   │   ├── Codigo_Estacao, Nome_Estacao, Lat, Lon, Alt, Tipo
│   │   ├── Data, Valor_Original, Valor_Preenchido
│   │   └── Flag_Origem (ORIGINAL / IDW / MEDIA_MENSAL / NAO_PREENCHIVEL)
│   │
│   ├── A002_preenchido.csv
│   └── ... (813 estações)
│
├── Estacoes_Finalizadas_Decendios/
│   ├── TIF_IDW_Decendios/
│   │   ├── Precip_Decendio_2023_01.tif
│   │   ├── Precip_Decendio_2023_02.tif
│   │   └── ... (108 TIFs)
│   │
│   ├── PNG_IDW_Decendios/ (se solicitado)
│   │   ├── Precip_Decendio_2023_01.png
│   │   └── ...
│   │
│   ├── Resumo_IDW_Decendios.json
│   │   ├── _legenda (explicação de todas as colunas)
│   │   ├── decendios (estatísticas por período)
│   │   └── cobertura (% do grid com dados)
│   │
│   └── Relatorio_Decendios.csv
│       ├── Estacao, Total_Decendios, Usaveis, Descartados
│       └── Percentual_Usavel, Status_Geral
│
└── Validacao_Decendios.csv
    └── Registro de todas as estações e qualidade
```

---

## Dicas e Boas Práticas

### 1. Escolha do Número de Vizinhos
- **3-5:** Recomendado (equilibra qualidade e velocidade)
- **>10:** Só se a densidade de estações é muito baixa

### 2. Interpretação dos Flags
| Flag | Significado | Confiança |
|------|-------------|-----------|
| ORIGINAL | Dado do CSV original | 100% |
| IDW (< 50km) | Interpolado de vizino perto | 85-95% |
| IDW (50-100km) | Interpolado de vizino longe | 60-80% |
| MEDIA_MENSAL | Climatologia do período | 40-70% |
| NAO_PREENCHIVEL | Impossível preencher | 0% (descarta) |

### 3. Validação de Qualidade
- Decêndios com >2 dias de flag ruim são **marcados como DESCARTADOS**
- No IDW final, estações marcadas como DESCARTADAS são **ignoradas**
- Permitindo que apenas dados de boa qualidade alimentem a interpolação

### 4. Performance
- 813 estações × 108 decêndios = ~88k registros
- Interpolação IDW: ~3-5 minutos no total
- TIFs com compressão LZW: ~15-20 MB cada

